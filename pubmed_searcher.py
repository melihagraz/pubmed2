import streamlit as st
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import io
import requests
import xml.etree.ElementTree as ET
import logging
import json
import re
from urllib.parse import quote

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SimplePubMedSearcher:
    """
    Production-ready PubMed searcher optimized for gene/biomarker literature research.
    Features:
    - Smart query construction for better results
    - Robust error handling and recovery
    - Evidence scoring based on multiple factors
    - Efficient rate limiting and caching
    """
    
    def __init__(self, email: str, api_key: Optional[str] = None, delay: float = 0.34):
        """
        Initialize PubMed searcher.
        
        Args:
            email: Valid email for NCBI API (required)
            api_key: NCBI API key for higher rate limits (optional)
            delay: Delay between requests in seconds
        """
        self.email = email
        self.api_key = api_key
        # Use faster rate with API key, slower without
        self.delay = 0.1 if api_key else max(delay, 0.34)
        self.last_request = 0
        self.cache = {}
        self.article_cache = {}
        
        # Enhanced search parameters
        self.max_retries = 3
        self.timeout = 15
        
        # Gene symbol patterns for validation
        self.gene_pattern = re.compile(r'^[A-Z][A-Z0-9-]*[0-9]*$')
        
    def _normalize_gene_name(self, gene_name: str) -> str:
        """Normalize gene names for better search results."""
        # Clean and standardize gene names
        gene = gene_name.strip().upper()
        
        # Remove common prefixes/suffixes that interfere with search
        gene = re.sub(r'_HUMAN$', '', gene)
        gene = re.sub(r'_EXPR$', '', gene)
        gene = re.sub(r'_LEVEL$', '', gene)
        
        # Handle common gene name patterns
        if '_' in gene and len(gene.split('_')) == 2:
            parts = gene.split('_')
            if parts[1].isdigit():
                gene = f"{parts[0]}{parts[1]}"  # Convert GENE_1 to GENE1
                
        return gene
    
    def _build_smart_query(self, feature_name: str, disease_context: str = None) -> str:
        """
        Build intelligent search query with fallback strategies.
        Uses progressive query simplification for better results.
        """
        gene = self._normalize_gene_name(feature_name)
        
        # Strategy 1: Specific biomarker query
        base_terms = [f'"{gene}"[Title/Abstract]']
        
        # Add disease context if provided
        if disease_context and disease_context.strip():
            disease = disease_context.strip().lower()
            base_terms.append(f'"{disease}"[Title/Abstract]')
        
        # Core biomedical terms (use OR to broaden results)
        bio_terms = [
            "biomarker[Title/Abstract]",
            "gene expression[Title/Abstract]", 
            "protein expression[Title/Abstract]",
            "clinical significance[Title/Abstract]",
            "prognostic[Title/Abstract]"
        ]
        
        # Construct progressive queries (from specific to general)
        queries = []
        
        # Query 1: Most specific
        if disease_context:
            specific_query = f"({base_terms[0]}) AND ({base_terms[1]}) AND ({' OR '.join(bio_terms[:2])})"
            queries.append(specific_query)
        
        # Query 2: Medium specificity
        medium_query = f"({base_terms[0]}) AND ({' OR '.join(bio_terms[:3])})"
        queries.append(medium_query)
        
        # Query 3: Broad search
        broad_query = f"({base_terms[0]}) AND (biomarker[Title/Abstract] OR expression[Title/Abstract])"
        queries.append(broad_query)
        
        # Query 4: Simple gene search
        simple_query = f'"{gene}"[Title/Abstract]'
        queries.append(simple_query)
        
        return queries[0]  # Start with most specific
    
    def _rate_limit(self):
        """Smart rate limiting with jitter to avoid burst requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request
        
        if time_since_last < self.delay:
            sleep_time = self.delay - time_since_last
            # Add small jitter to avoid synchronized requests
            sleep_time += 0.05 * (hash(str(current_time)) % 10) / 10
            time.sleep(sleep_time)
            
        self.last_request = time.time()
    
    def _make_request(self, url: str, params: dict, max_retries: int = None) -> requests.Response:
        """
        Make HTTP request with retry logic and smart error handling.
        """
        if max_retries is None:
            max_retries = self.max_retries
            
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                
                response = requests.get(url, params=params, timeout=self.timeout)
                
                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", "2"))
                    logging.warning(f"Rate limited, waiting {retry_after} seconds")
                    time.sleep(retry_after + 1)
                    continue
                
                # Handle server errors with exponential backoff
                if response.status_code >= 500:
                    wait_time = (2 ** attempt) + 1
                    logging.warning(f"Server error {response.status_code}, retrying in {wait_time}s")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + 1
                    logging.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                    continue
                    
        # All retries failed
        raise last_exception
    
    def _search_with_fallback(self, feature_name: str, disease_context: str = None, max_results: int = 5) -> dict:
        """
        Search with progressive query fallback for better results.
        """
        gene = self._normalize_gene_name(feature_name)
        
        # Generate multiple query strategies
        queries = []
        
        # Strategy 1: Disease-specific biomarker search
        if disease_context:
            disease = disease_context.strip().lower()
            queries.append({
                'query': f'"{gene}"[Title/Abstract] AND "{disease}"[Title/Abstract] AND biomarker[Title/Abstract]',
                'strategy': 'disease_biomarker'
            })
            
        # Strategy 2: General biomarker search  
        queries.append({
            'query': f'"{gene}"[Title/Abstract] AND (biomarker[Title/Abstract] OR "gene expression"[Title/Abstract])',
            'strategy': 'general_biomarker'
        })
        
        # Strategy 3: Expression studies
        queries.append({
            'query': f'"{gene}"[Title/Abstract] AND expression[Title/Abstract]',
            'strategy': 'expression'
        })
        
        # Strategy 4: Simple gene search
        queries.append({
            'query': f'"{gene}"[Title/Abstract]',
            'strategy': 'simple'
        })
        
        # Try each query until we get sufficient results
        for query_info in queries:
            try:
                result = self._execute_search(query_info['query'], max_results)
                
                if result.get('count', 0) > 0:
                    result['search_strategy'] = query_info['strategy'] 
                    result['search_query'] = query_info['query']
                    return result
                    
            except Exception as e:
                logging.warning(f"Query failed: {query_info['strategy']} - {e}")
                continue
        
        # All queries failed
        return {
            'count': 0,
            'ids': [],
            'search_strategy': 'all_failed',
            'search_query': queries[0]['query'] if queries else ''
        }
    
    def _execute_search(self, query: str, max_results: int) -> dict:
        """Execute single search query."""
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "email": self.email,
            "tool": "FeatureSelectionAgent",
            "sort": "relevance",
            "datetype": "pdat",
            "reldate": "3650"  # Last 10 years for relevance
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        response = self._make_request(search_url, params)
        data = response.json()
        
        esearch_result = data.get("esearchresult", {})
        return {
            'count': int(esearch_result.get("count", "0")),
            'ids': esearch_result.get("idlist", [])
        }
    
    def _calculate_evidence_score(self, paper_count: int, disease_context: str = None, 
                                 articles: List[dict] = None, search_strategy: str = None) -> float:
        """
        Advanced evidence scoring algorithm.
        Considers multiple factors for more accurate literature support assessment.
        """
        if paper_count == 0:
            return 0.0
            
        # Base score from paper count (logarithmic scale)
        import math
        base_score = min(math.log10(paper_count + 1) * 1.5, 4.0)
        
        # Bonus factors
        bonuses = 0.0
        
        # Disease context bonus
        if disease_context and paper_count > 0:
            bonuses += 0.3
            
        # Search strategy bonus (more specific strategies get higher scores)
        strategy_bonuses = {
            'disease_biomarker': 0.4,
            'general_biomarker': 0.2,
            'expression': 0.1,
            'simple': 0.0
        }
        bonuses += strategy_bonuses.get(search_strategy, 0.0)
        
        # Article quality bonus (if articles were fetched)
        if articles:
            recent_articles = sum(1 for a in articles if self._is_recent_article(a))
            if recent_articles > 0:
                bonuses += 0.2 * (recent_articles / len(articles))
        
        # Paper count thresholds for quality assessment
        if paper_count >= 100:
            bonuses += 0.3  # Well-studied gene
        elif paper_count >= 50:
            bonuses += 0.2
        elif paper_count >= 20:
            bonuses += 0.1
            
        final_score = min(base_score + bonuses, 5.0)
        return round(final_score, 1)
    
    def _is_recent_article(self, article: dict) -> bool:
        """Check if article is from recent years (last 5 years)."""
        try:
            year = int(article.get('year', '0'))
            current_year = 2025  # Update this as needed
            return year >= (current_year - 5)
        except:
            return False
    
    def fetch_article_details(self, pmids: List[str]) -> List[dict]:
        """
        Fetch detailed article information with robust error handling.
        """
        if not pmids:
            return []
            
        articles = []
        
        try:
            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
                "email": self.email,
                "tool": "FeatureSelectionAgent",
            }
            
            if self.api_key:
                params["api_key"] = self.api_key
            
            response = self._make_request(fetch_url, params)
            
            # Validate XML content
            content_type = response.headers.get("Content-Type", "").lower()
            if "xml" not in content_type:
                logging.warning(f"Unexpected content type: {content_type}")
                return []
            
            # Parse XML
            root = ET.fromstring(response.content)
            
            for article_elem in root.findall(".//PubmedArticle"):
                try:
                    article = self._parse_article_robust(article_elem)
                    if article:
                        articles.append(article)
                except Exception as e:
                    logging.warning(f"Failed to parse article: {e}")
                    continue
                    
        except Exception as e:
            logging.error(f"Failed to fetch article details: {e}")
            
        return articles
    
    def _parse_article_robust(self, article_elem) -> Optional[dict]:
        """
        Robust article parsing with multiple fallback strategies.
        """
        try:
            # Extract PMID
            pmid = self._extract_text(article_elem, './/PMID', default="Unknown")
            
            # Extract title with fallbacks
            title = self._extract_text(article_elem, './/ArticleTitle', default="No title available")
            if not title or title == "No title available":
                title = self._extract_text(article_elem, './/BookTitle', default="No title available")
            
            # Extract authors
            authors = self._extract_authors(article_elem)
            
            # Extract journal
            journal = self._extract_journal(article_elem)
            
            # Extract year
            year = self._extract_year(article_elem)
            
            # Extract abstract
            abstract = self._extract_abstract(article_elem)
            
            return {
                'pmid': pmid,
                'title': title,
                'authors': authors,
                'journal': journal,
                'year': year,
                'abstract': abstract,
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            }
            
        except Exception as e:
            logging.error(f"Error parsing article: {e}")
            return None
    
    def _extract_text(self, element, xpath: str, default: str = "") -> str:
        """Safely extract text from XML element."""
        try:
            elem = element.find(xpath)
            if elem is not None:
                # Handle both direct text and nested elements
                text = elem.text or ''.join(elem.itertext())
                return text.strip() if text else default
        except:
            pass
        return default
    
    def _extract_authors(self, article_elem) -> str:
        """Extract and format author list."""
        authors = []
        try:
            for author in article_elem.findall('.//Author'):
                last_name = self._extract_text(author, 'LastName')
                fore_name = self._extract_text(author, 'ForeName')
                
                if last_name:
                    if fore_name:
                        # Use first initial only
                        initial = fore_name[0] if fore_name else ""
                        authors.append(f"{last_name} {initial}")
                    else:
                        authors.append(last_name)
                        
            if authors:
                author_str = ", ".join(authors[:3])
                if len(authors) > 3:
                    author_str += " et al."
                return author_str
                
        except Exception:
            pass
            
        return "Authors not available"
    
    def _extract_journal(self, article_elem) -> str:
        """Extract journal name with fallbacks."""
        # Try multiple journal name fields
        for xpath in ['.//Journal/Title', './/Journal/ISOAbbreviation', './/MedlineTA']:
            journal = self._extract_text(article_elem, xpath)
            if journal and journal != "":
                return journal
        return "Unknown Journal"
    
    def _extract_year(self, article_elem) -> str:
        """Extract publication year with fallbacks."""
        # Try multiple date fields
        for xpath in ['.//PubDate/Year', './/PubDate/MedlineDate']:
            date_str = self._extract_text(article_elem, xpath)
            if date_str:
                # Extract year from various date formats
                year_match = re.search(r'(\d{4})', date_str)
                if year_match:
                    return year_match.group(1)
        return "Unknown"
    
    def _extract_abstract(self, article_elem) -> str:
        """Extract abstract with length control."""
        try:
            # Try structured abstract first
            abstract_parts = []
            for abstract_elem in article_elem.findall('.//AbstractText'):
                label = abstract_elem.get('Label', '')
                text = abstract_elem.text or ''.join(abstract_elem.itertext())
                
                if text:
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
            
            if abstract_parts:
                abstract = ' '.join(abstract_parts)
            else:
                # Try simple abstract
                abstract = self._extract_text(article_elem, './/AbstractText')
            
            if abstract and len(abstract) > 600:
                abstract = abstract[:597] + "..."
                
            return abstract if abstract else "No abstract available"
            
        except Exception:
            return "No abstract available"
    
    def search_simple(self, feature_name: str, disease_context: str = None, max_results: int = 5) -> dict:
        """
        Main search interface with comprehensive error handling and caching.
        """
        # Create cache key
        cache_key = f"{feature_name}_{disease_context}_{max_results}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Perform search with fallback
            search_result = self._search_with_fallback(feature_name, disease_context, max_results)
            
            count = search_result['count']
            ids = search_result['ids']
            
            # Fetch article details for top results
            articles = []
            if ids:
                try:
                    articles = self.fetch_article_details(ids[:5])
                except Exception as e:
                    logging.warning(f"Failed to fetch articles for {feature_name}: {e}")
            
            # Calculate evidence score
            evidence_score = self._calculate_evidence_score(
                count, 
                disease_context, 
                articles, 
                search_result.get('search_strategy')
            )
            
            result = {
                'feature_name': feature_name,
                'paper_count': count,
                'sample_ids': ids[:5],
                'articles': articles,
                'evidence_score': evidence_score,
                'search_query': search_result.get('search_query', ''),
                'search_strategy': search_result.get('search_strategy', 'unknown'),
                'disease_context': disease_context
            }
            
            # Cache successful results
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            error_msg = f"Search failed for {feature_name}: {str(e)}"
            logging.error(error_msg)
            
            return {
                'feature_name': feature_name,
                'paper_count': 0,
                'sample_ids': [],
                'articles': [],
                'evidence_score': 0.0,
                'search_query': '',
                'disease_context': disease_context,
                'error': str(e)
            }
    
    def batch_search(self, features: List[str], disease_context: str = None, 
                    progress_callback=None) -> List[dict]:
        """
        Efficient batch search with progress tracking and smart pacing.
        """
        results = []
        total = len(features)
        successful = 0
        
        logging.info(f"Starting batch search for {total} features")
        
        for i, feature in enumerate(features):
            try:
                if progress_callback:
                    progress_callback(i + 1, total, feature)
                
                result = self.search_simple(feature, disease_context)
                results.append(result)
                
                if result.get('paper_count', 0) > 0:
                    successful += 1
                
                # Progress logging
                if (i + 1) % 5 == 0:
                    logging.info(f"Batch progress: {i+1}/{total} complete, {successful} successful")
                    
            except Exception as e:
                logging.error(f"Batch search error for {feature}: {e}")
                results.append({
                    'feature_name': feature,
                    'paper_count': 0,
                    'sample_ids': [],
                    'articles': [],
                    'evidence_score': 0.0,
                    'search_query': '',
                    'disease_context': disease_context,
                    'error': str(e)
                })
        
        logging.info(f"Batch search complete: {successful}/{total} successful searches")
        return results