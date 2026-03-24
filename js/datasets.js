// No dataset selector in the UI for now.
// This file exists so adding one later is a small, pleasant diff.

export const DATASETS = {
  default: {
    name: "default",
    embeddingsUrl: "./data/embeddings.json",
    classWsUrl: "./data/class_ws.json",
  },
  clay: {
    name: "Clay land cover",
    embeddingsUrl: "./data/clay_embeddings.json",
    classWsUrl: "./data/clay_class_ws.json",
    dim: 1024,
  },
};

export const DEFAULT_DATASET_KEY = "default";
