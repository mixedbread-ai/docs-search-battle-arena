import {
  SearchProvider,
  SearchResult,
  MixedBreadSearchCredentials,
} from "./types";
import slugify from "slugify";
import Mixedbread from "@mixedbread/sdk";

interface Heading {
  text: string;
  level: number;
}

interface ChunkMetadata {
  title?: string;
  path?: string;
  heading_context?: Heading[];
  chunk_headings?: Heading[];
}

export class MixedBreadSearchProvider implements SearchProvider {
  private credentials: MixedBreadSearchCredentials;
  name = "mxbai_search";

  constructor(credentials: MixedBreadSearchCredentials) {
    this.credentials = credentials;
  }

  async search(query: string): Promise<SearchResult[]> {
    try {
      // Initialize the MXBAI Search client
      const mxbai = new Mixedbread({
        apiKey: this.credentials.apiKey ?? "",
      });

      const res = await mxbai.stores.search({
        query,
        store_identifiers: [this.credentials.storeId],
        top_k: 10,
        search_options: {
          return_metadata: true,
          rerank: this.credentials.reranking,
        },
      });

      const structuredResponse = res.data.flatMap((item, index) => {
        const metadata = (item.generated_metadata ?? {}) as ChunkMetadata;
        const description = "text" in item ? item.text : "";
        const headingContext = Array.isArray(metadata.heading_context)
          ? metadata.heading_context
          : [];
        const chunkHeadings = Array.isArray(metadata.chunk_headings)
          ? metadata.chunk_headings
          : [];
        const pageTitle =
          metadata.title ||
          headingContext.find((h) => h.level === 1)?.text ||
          chunkHeadings.find((h) => h.level === 1)?.text ||
          "Untitled";

        // Get section_title: first level 2 heading from chunk_headings, then heading_context
        const secondaryTitle =
          chunkHeadings.find(
            (h: { text: string; level: number }) => h.level === 2,
          )?.text || "";
        const anchor = slugify(secondaryTitle, { lower: true });
        const sectionTitle =
          chunkHeadings.find(
            (h: { text: string; level: number }) => h.level === 2,
          )?.text ||
          headingContext.find(
            (h: { text: string; level: number }) => h.level === 2,
          )?.text ||
          "";

        let url = "";
        if (metadata.path) {
          url = anchor !== "" ? `${metadata.path}#${anchor}` : metadata.path;
        } else if (item.filename) {
          // Extract path starting from /docs/ and remove file extension
          const docsIndex = item.filename.indexOf("/docs/");
          if (docsIndex !== -1) {
            const pathFromDocs = item.filename.substring(docsIndex);
            // Remove file extension (.md, .mdx, etc.)
            url = pathFromDocs.replace(/\.[^/.]+$/, "");
          }
        }

        return [
          {
            id: `${item.file_id}-${index}-page`,
            title: pageTitle,
            description: description ?? "",
            url,
            score: item.score,
          },
        ];
      });

      return structuredResponse;
    } catch (error) {
      console.error("Error searching MXBAI:", error);
      throw new Error(
        `MXBAI search failed: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }
}
