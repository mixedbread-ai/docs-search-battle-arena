import { SearchResult } from "../providers";
import { generateText } from "ai";

interface EvaluationResult {
  score: number;
  feedback: string;
  metrics?: {
    ndcg5: number;
    ndcg10: number;
    nEU: number;
  };
}

export class LLMService {
  private modelName = "google/gemini-2.5-flash";
  private hasApiKey: boolean;

  private static readonly MAX_PASSAGE_CHARS = 4000;
  private static readonly MAX_TITLE_CHARS = 200;

  // NEW: single place to control evaluation depth and EU alpha
  private static readonly EVAL_DEPTH = 10;
  private static readonly EU_ALPHA = 0.9;

  constructor() {
    const key =
      process.env.AI_GATEWAY_API_KEY ||
      "";

    if (process.env.AI_GATEWAY_API_KEY) {
      process.env.GOOGLE_GENERATIVE_AI_API_KEY = process.env.AI_GATEWAY_API_KEY;
    }

    this.hasApiKey = !!key;
  }

  /**
   * Evaluates search results and assigns a score using Gemini 2.5 Flash
   * If no API key is available, returns fallback scores of 0
   */
  async evaluateSearchResults(
    query: string,
    results1: SearchResult[],
    results2: SearchResult[]
  ): Promise<{
    db1: EvaluationResult;
    db2: EvaluationResult;
    llmDuration: number;
  }> {
    if (!this.hasApiKey) {
      console.log("No LLM API key available, returning fallback scores");
      return this.fallback();
    }

    // NEW: cap to top-10 before scoring
    const top1 = results1.slice(0, LLMService.EVAL_DEPTH);
    const top2 = results2.slice(0, LLMService.EVAL_DEPTH);

    // Score each dataset (top-k only)
    const d1 = await this.scoreDataset(query, top1);
    const d2 = await this.scoreDataset(query, top2);
    const llmDuration = d1.durationMs + d2.durationMs;

    // NEW: build arrays of per-rank labels for metrics
    const s1 = (d1.perDoc || [])
      .sort((a, b) => a.index - b.index)
      .map(x => x.score)
      .slice(0, LLMService.EVAL_DEPTH);

    const s2 = (d2.perDoc || [])
      .sort((a, b) => a.index - b.index)
      .map(x => x.score)
      .slice(0, LLMService.EVAL_DEPTH);

    // NEW: pooled labels for ideal ranking (union of both top-10 lists)
    const pooled = [...s1, ...s2];

    const m1 = this.computeMetrics(s1, pooled);
    const m2 = this.computeMetrics(s2, pooled);

    return {
      db1: {
        score: m1.nEU,
        feedback: this.makeFeedback(d1, 1, m1), // NEW: include printed metrics
      },
      db2: {
        score: m2.nEU,
        feedback: this.makeFeedback(d2, 2, m2), // NEW
      },
      llmDuration,
    };
  }

  private fallback() {
    return {
      db1: { score: -1, feedback: "" },
      db2: { score: -1, feedback: "" },
      llmDuration: 0,
    };
  }

  private truncate(s: string, max: number) {
    if (!s) return "";
    return s.length <= max ? s : s.slice(0, max);
  }

  private formatPassage(r: SearchResult) {
    const title = this.truncate(r.title || "Untitled", LLMService.MAX_TITLE_CHARS);
    const desc = this.truncate(r.description || "", LLMService.MAX_PASSAGE_CHARS);
    const url = r.url ? `\nURL: ${r.url}` : "";
    return `Title: ${title}\n Description: ${desc}\n URL: ${url}`.trim();
  }

  private umbrelaPrompt(query: string, passage: string) {
    return (
      `Given a query and a passage, you must provide a score on an integer scale of 0 to 3 with the following meanings:\n` +
      `0 = represent that the passage has nothing to do with the query, \n` +
      `1 = represents that the passage seems related to the query but does not answer it, \n` +
      `2 = represents that the passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information and \n` +
      `3 = represents that the passage is dedicated to the query and contains the exact answer.\n\n` +
      `Important Instruction: Assign category 1 if the passage is somewhat related to the topic but not completely, category 2 if passage presents something very important related to the entire topic but also has some extra information and category 3 if the passage only and entirely refers to the topic. If none of the above satisfies give it category 0.\n\n` +
      `Query: ${query}\n` +
      `Passage: ${passage}\n\n` +
      `Split this problem into steps:\n` +
      `Consider the underlying intent of the search.\n` +
      `Measure how well the content matches a likely intent of the query (M).\n` +
      `Measure how trustworthy the passage is (T).\n` +
      `Consider the aspects above and the relative importance of each, and decide on a final score (O). Final score must be an integer value only.\n` +
      `Do not provide any code in result. Provide each score in the format of: ##final score: score without providing any reasoning.\n`
    );
  }

  private extractScore(rawText: string): number {
    // Accept formats like "##final score: 2"; be lenient to stray whitespace/casing
    const match = /##\s*final\s*score\s*:\s*([0-3])\b/i.exec(rawText || "");
    if (!match) return -1; // indicates parse error; caller will treat as 0
    const n = parseInt(match[1], 10);
    if (Number.isNaN(n)) return -1;
    if (n < 0 || n > 3) return -1;
    return n;
  }

  private async scoreOne(query: string, passage: string): Promise<{ score: number; raw: string; durationMs: number; }> {
    const prompt = this.umbrelaPrompt(query, passage);

    const start = (typeof performance !== "undefined" ? performance : { now: () => Date.now() }).now();
    let text = "";
    try {
      const result = await generateText({
        model: this.modelName,
        prompt,
        temperature: 0.0,
      });
      text = (result as any)?.text || "";
    } catch (err) {
      console.log("err", err);
      // On per-chunk failure, return score 0 and bubble raw error for feedback aggregation
      const endErr = (typeof performance !== "undefined" ? performance : { now: () => Date.now() }).now();
      return { score: 0, raw: `ERR:${String(err)}`, durationMs: endErr - start };
    }
    const end = (typeof performance !== "undefined" ? performance : { now: () => Date.now() }).now();

    let score = this.extractScore(text);
    if (score === -1) {
      score = 0;
    }

    return { score, raw: text, durationMs: end - start };
  }

  private async scoreDataset(
    query: string,
    results: SearchResult[]
  ): Promise<{
    total: number;
    counts: Record<0 | 1 | 2 | 3, number>;
    errors: number;
    durationMs: number;
    perDoc?: Array<{ index: number; score: number }>;
  }> {
    let total = 0;
    const counts: Record<0 | 1 | 2 | 3, number> = { 0: 0, 1: 0, 2: 0, 3: 0 };
    let errors = 0;
    let durationMs = 0;

    const perDoc: Array<{ index: number; score: number }> = [];

    const capped = results.slice(0, LLMService.EVAL_DEPTH);

    for (let i = 0; i < capped.length; i++) {
      const r = capped[i];
      const passage = this.truncate(this.formatPassage(r), LLMService.MAX_PASSAGE_CHARS);
      const { score, raw, durationMs: d } = await this.scoreOne(query, passage);
      durationMs += d;

      if (raw.startsWith("ERR:") || !/^##/m.test(raw)) {
        if (raw.startsWith("ERR:")) errors += 1;
      }

      const s = (score < 0 || score > 3) ? 0 : (score as 0 | 1 | 2 | 3);
      counts[s] = (counts[s] || 0) + 1;
      total += s;
      perDoc.push({ index: i, score: s });
    }

    return { total, counts, errors, durationMs, perDoc };
  }


  private gain(rel: number): number {
    return Math.pow(2, rel) - 1;
  }

  private dcgAt(scores: number[], D: number): number {
    let sum = 0;
    const L = Math.min(D, scores.length);
    for (let i = 0; i < L; i++) {
      const rel = scores[i] ?? 0;
      sum += this.gain(rel) / Math.log2(i + 2);
    }
    return sum;
  }

  private ndcgAt(scores: number[], idealPool: number[], D: number): number {
    const ideal = idealPool.slice().sort((a, b) => b - a);
    const dcgSys = this.dcgAt(scores, D);
    const dcgIdeal = this.dcgAt(ideal, D);
    return dcgIdeal > 0 ? dcgSys / dcgIdeal : 0;
  }

  private expectedUtilityOutOf10(
    scores: number[],
    alpha = LLMService.EU_ALPHA,
    L = LLMService.EVAL_DEPTH
  ): number {
    let num = 0;
    let denomP = 0;
    for (let i = 0; i < L; i++) {
      const p = Math.pow(alpha, i);
      const rel = scores[i] ?? 0;
      num += p * this.gain(rel);
      denomP += p;
    }
    // normalize by max gain = 7 at every position
    const denom = denomP * this.gain(3);
    return denom > 0 ? (num / denom) * 10 : 0;
  }
  private computeMetrics(scores: number[], pooled: number[]) {
    return {
      // ndcg5: this.ndcgAt(scores, pooled, 5),
      // ndcg10: this.ndcgAt(scores, pooled, 10),
      nEU: this.expectedUtilityOutOf10(scores, LLMService.EU_ALPHA, LLMService.EVAL_DEPTH),
    };
  }


  private makeFeedback(
    agg: { total: number; counts: Record<0 | 1 | 2 | 3, number>; errors: number; perDoc?: Array<{ index: number; score: number }>; },
    dbIndex: 1 | 2,
    metrics?: { ndcg5: number; ndcg10: number; nEU: number } // NEW
  ) {
    const n = Object.values(agg.counts).reduce((a, b) => a + b, 0);
    const parts: string[] = [];
    parts.push(`DB ${dbIndex}: ${n} docs scored (top ${LLMService.EVAL_DEPTH}). Total label sum: ${agg.total}.`);
    parts.push(`Breakdown — 3★: ${agg.counts[3]}, 2★: ${agg.counts[2]}, 1★: ${agg.counts[1]}, 0★: ${agg.counts[0]}.`);
    if (metrics) {
      parts.push(
        `Score: ${metrics.nEU.toFixed(3)}.`
      );
    }
    if (agg.errors > 0) parts.push(`Note: ${agg.errors} chunk(s) failed to parse or errored and were counted as 0.`);
    return parts.join(" ");
  }
}