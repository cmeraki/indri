import arxiv
import time
from datetime import datetime, timedelta

from omni.logger import get_logger

logger = get_logger(__name__)

def get_recent_ai_papers():
    twelve_hours_ago = datetime.utcnow() - timedelta(hours=12)

    client = arxiv.Client()

    # Create a search query for AI papers
    search = arxiv.Search(
        query="cat:cs.AI OR cat:stat.ML",
        max_results=100,  # Adjust this number as needed
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    recent_papers = []

    try:
        for result in client.results(search):
            # # Stop if we've reached papers older than 12 hours
            # if result.published < twelve_hours_ago:
            #     break

            paper_info = {
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'abstract': result.summary,
                'published': result.published,
                'url': result.entry_id,
                'categories': result.categories
            }
            recent_papers.append(paper_info)

            # Add a small delay to avoid overwhelming the server
            time.sleep(0.1)

    except Exception as e:
        print(f"An error occurred while fetching results: {e}")

    return recent_papers

if __name__ == "__main__":
    papers = get_recent_ai_papers()

    print(f"Found {len(papers)} AI papers in the last 12 hours:")

    for paper in papers:
        print(f"\nTitle: {paper['title']}")
        print(f"Authors: {', '.join(paper['authors'])}")
        print(f"Abstract: {paper['abstract'][:200]}...")  # Print first 200 characters of abstract
        print(f"Published: {paper['published']}")
        print(f"URL: {paper['url']}")
        print(f"Categories: {', '.join(paper['categories'])}")