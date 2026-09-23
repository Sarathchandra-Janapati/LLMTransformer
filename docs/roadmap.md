# Product Roadmap

MiniGPT Studio grows in layers. Each milestone should run, be explainable, and
leave a useful interface for the next one.

## Milestone 0: Foundation

- Create the monorepo layout.
- Keep model code in Python packages isolated from the future API and UI.
- Add tests for the first NumPy math primitives.

## Milestone 1: Math Lab

- Implement dot product and matrix multiplication manually with NumPy arrays. Done.
- Implement a neuron, dense layer, activations, loss, gradient descent, and
  backpropagation. In progress with a small two-layer regression network.
- Expose small examples that can be visualized later.

## Milestone 2: Attention Lab

- Tokenize a fixed sentence. Done for a tiny whitespace example.
- Project embeddings into Q, K, and V. In progress with single-head
  self-attention tensors exposed for visualization after position encodings are
  added to token embeddings.
- Compute scaled dot-product attention and causal masking separately. Done for
  the single-head attention lab.
- Build the first React heatmap and token inspector. In progress with editable
  lab sentences, learning task modes, masked-vs-unmasked comparison, a guided
  backend pipeline, projection math, formulas, variable meanings, and worked
  attention-score math rendered in the frontend.

## Milestone 3: Transformer Core

- Add multi-head attention. Done at the core NumPy level with inspectable
  per-head results.
- Add layer normalization. Done at the core NumPy level.
- Add feed-forward layers. Done at the core NumPy level.
- Add residual connections and the first pre-norm transformer block. Done at
  the core NumPy level.
- Add decoder stack and language-model prediction head. Done at the tiny NumPy
  core level.
- Add token embeddings plus positional encodings around the decoder so the tiny
  GPT path can start from token ids. Done with an inspectable forward wrapper.
- Start next-token training. Done with sliding context windows, cross-entropy,
  accuracy, and gradient updates for the final vocabulary projection while the
  decoder remains frozen.
- Train token embeddings. Done with tiny finite-difference gradients so learners
  can see how embedding changes affect next-token loss before full backprop.
- Add analytic backprop building blocks. Done for LayerNorm and the transformer
  feed-forward network with finite-difference gradient checks.
- Wire transformer-block backprop. Done for residual flow, feed-forward,
  LayerNorm, multi-head attention, softmax, and Q/K/V projection gradients.
- Wire decoder-stack and full-model training. Done with analytic backprop from
  cross-entropy logits through the LM head, decoder blocks, attention, MLPs,
  norms, and token embeddings.
- Expose training through the backend. Done with FastAPI endpoints for default
  config, running tiny GPT training, and reading the latest run artifact.
- Build the first training dashboard. Done with real backend-driven controls,
  loss curve, predictions, embedding table, and projection-weight table.
- Add Kaggle-style training progress. Done with epoch-by-epoch step logs,
  progress bars, loss, average loss, and accuracy in the dashboard.
- Train a tiny model on a tiny corpus before adding service complexity.


## Later milestones

- Training dashboard and sampling comparison UI.
- Fine-tuning experiments with LoRA-style adapters.
- RAG ingestion and chat over documents.
- FastAPI services, auth, streaming, caching, containers, and deployment.
