const VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings";
const VOYAGE_MODEL = "voyage-3.5";

export async function embedQuery(text) {
  const response = await fetch(VOYAGE_API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.VOYAGE_API_KEY}`,
    },
    body: JSON.stringify({ input: [text], model: VOYAGE_MODEL, input_type: "query" }),
  });
  if (!response.ok) throw new Error(`Voyage embed failed: ${response.status}`);
  const data = await response.json();
  return data.data[0].embedding;
}

export async function searchKnowledgeBase(queryEmbedding, topK = 5, kbIds = null) {
  const body = { vector: queryEmbedding, topK, includeMetadata: true };
  if (kbIds && kbIds.length > 0) body.filter = { kbId: { $in: kbIds } };
  const response = await fetch(`${process.env.PINECONE_INDEX_HOST}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Api-Key": process.env.PINECONE_API_KEY },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`Pinecone query failed: ${response.status}`);
  const data = await response.json();
  return data.matches || [];
}
