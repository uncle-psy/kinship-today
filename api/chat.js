import { embedQuery, searchKnowledgeBase } from "./_rag.js";

const DEFAULT_SYSTEM_PROMPT = `You are a helpful, friendly assistant. You speak clearly and warmly, and keep your responses conversational — usually 2-4 paragraphs unless the person needs more depth.`;

const chatbotCache = new Map();
const CACHE_TTL = 60_000;

function safeParse(json, fallback) {
  try { return JSON.parse(json); } catch { return fallback; }
}

async function getChatbotConfig(chatbotId) {
  if (!chatbotId) return null;
  const cached = chatbotCache.get(chatbotId);
  if (cached && (Date.now() - cached.ts) < CACHE_TTL) return cached.data;
  try {
    if (!process.env.PINECONE_API_KEY || !process.env.PINECONE_INDEX_HOST) return null;
    const url = `${process.env.PINECONE_INDEX_HOST}/vectors/fetch?ids=${chatbotId}`;
    const response = await fetch(url, { headers: { "Api-Key": process.env.PINECONE_API_KEY } });
    const data = await response.json();
    const meta = data.vectors?.[chatbotId]?.metadata;
    if (meta?.systemPrompt) {
      const config = { systemPrompt: meta.systemPrompt, kbIds: safeParse(meta.kbIdsJson, []) };
      chatbotCache.set(chatbotId, { data: config, ts: Date.now() });
      return config;
    }
  } catch (err) { console.error("Chatbot config fetch error (non-fatal):", err.message); }
  return null;
}

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  const { message, history = [], chatbotId } = req.body;
  if (!message || typeof message !== "string") return res.status(400).json({ error: "Message is required" });
  if (!process.env.ANTHROPIC_API_KEY) return res.status(500).json({ error: "ANTHROPIC_API_KEY is not configured" });

  const config = await getChatbotConfig(chatbotId);
  const systemPrompt = config?.systemPrompt || DEFAULT_SYSTEM_PROMPT;
  const kbIds = config?.kbIds || [];

  let contextBlock = "";
  try {
    if (process.env.VOYAGE_API_KEY && process.env.PINECONE_API_KEY && process.env.PINECONE_INDEX_HOST) {
      const queryEmbedding = await embedQuery(message);
      const matches = await searchKnowledgeBase(queryEmbedding, 5, kbIds.length > 0 ? kbIds : null);
      const relevantChunks = matches.filter((m) => m.score >= 0.3).map((m) => m.metadata.text);
      if (relevantChunks.length > 0) {
        contextBlock = "\n\n<knowledge_base>\nThe following passages are from the knowledge base. Use them to inform your responses when relevant, but speak naturally — do not quote them verbatim or mention that you are referencing documents.\n\n" +
          relevantChunks.map((chunk, i) => `[${i + 1}] ${chunk}`).join("\n\n") + "\n</knowledge_base>";
      }
    }
  } catch (err) { console.error("RAG retrieval error (non-fatal):", err.message); }

  const messages = [...history.map((msg) => ({ role: msg.role, content: msg.content })), { role: "user", content: message }];

  try {
    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": process.env.ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: "claude-sonnet-4-20250514",
        max_tokens: 1024,
        system: systemPrompt + contextBlock,
        messages,
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      console.error("Anthropic API error:", data);
      return res.status(502).json({ error: "Something went wrong. Please try again." });
    }
    return res.status(200).json({ reply: data.content[0].text });
  } catch (error) {
    console.error("Anthropic API error:", error);
    return res.status(500).json({ error: "Something went wrong. Please try again." });
  }
}
