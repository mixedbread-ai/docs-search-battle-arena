import {
  SearchProvider,
  SearchResult,
  UpstashSearchCredentials,
} from "./types";
import { Search } from "@upstash/search";

export class UpstashSearchProvider implements SearchProvider {
  private credentials: UpstashSearchCredentials;
  name = "upstash_search";

  constructor(credentials: UpstashSearchCredentials) {
    this.credentials = credentials;
  }

  private deduplicateByUrl(results: SearchResult[]): SearchResult[] {
    const urlMap = new Map<string, SearchResult>();

    for (const result of results) {
      const url = result.url || "";
      const existing = urlMap.get(url);

      // Keep the result with the higher score, or the first one if scores are equal
      if (!existing || (result.score || 0) > (existing.score || 0)) {
        urlMap.set(url, result);
      }
    }

    // Return results in the order they first appeared
    const seenUrls = new Set<string>();
    return results.filter((result) => {
      const url = result.url || "";
      if (seenUrls.has(url)) {
        return false;
      }
      const mapResult = urlMap.get(url);
      if (mapResult === result) {
        seenUrls.add(url);
        return true;
      }
      return false;
    });
  }

  async search(query: string): Promise<SearchResult[]> {
    try {
      // Initialize the Upstash Search client
      const client = new Search({
        url: this.credentials.url,
        token: this.credentials.token,
      });

      // Access the specified index
      const index = client.index<{ title: string; description: string }>(
        this.credentials.index
      );

      // Perform the search
      const searchResults = await index.search({
        query,
        limit: 10,
        reranking: this.credentials.reranking,
        inputEnrichment: this.credentials.inputEnrichment,
      });

      // Transform Upstash search results to the common SearchResult format
      const transformedResults = searchResults.map((result) => {
        // Extract the document content and metadata
        const { id, content, score, metadata } = result;

        return {
          id,
          title: content.title ?? "Untitled",
          description: content.description ?? "No description available",
          url: metadata?.url ? `https://vercel.com${metadata.url}` : "No URL available",
          score: score || 0,
        };
      });

      // Deduplicate results by URL, keeping the highest scoring result for each URL
      return this.deduplicateByUrl(transformedResults);
    } catch (error) {
      console.error("Error searching Upstash:", error);
      throw new Error(
        `Upstash search failed: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }
}
