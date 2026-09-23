import { useEffect, useState } from "react";

const tensorViews = [
  { key: "embeddings", label: "Token Embeddings" },
  { key: "position_encodings", label: "Position Encodings" },
  { key: "attention_inputs", label: "Attention Inputs" },
  { key: "queries", label: "Queries" },
  { key: "keys", label: "Keys" },
  { key: "values", label: "Values" },
  { key: "context", label: "Context" },
];

const taskModeOptions = [
  { id: "next_token", label: "Next token" },
  { id: "fill_blank", label: "Fill blank" },
  { id: "classification", label: "Classify" },
  { id: "local_window", label: "Local" },
];

function formatValue(value) {
  return value === null ? "mask" : value.toFixed(3);
}

function formatVector(values) {
  return `[${values.map((value) => formatValue(value)).join(", ")}]`;
}

function formatMetric(value) {
  return Number(value).toFixed(4);
}

function dotTerms(left, right) {
  return left.map((value, index) => `${formatValue(value)} x ${formatValue(right[index])}`);
}

function Matrix({ columnLabels, matrix, rowLabels }) {
  return (
    <div
      className="matrix"
      style={{ "--column-count": matrix[0].length }}
    >
      <span className="matrix-corner" />
      {columnLabels.map((label, index) => (
        <span className="matrix-axis" key={`${label}-${index}`}>
          {label}
        </span>
      ))}
      {matrix.map((row, rowIndex) => (
        <div className="matrix-row" key={`${rowLabels[rowIndex]}-${rowIndex}`}>
          <span className="matrix-axis row-axis">{rowLabels[rowIndex]}</span>
          {row.map((value, columnIndex) => (
            <span className="matrix-value" key={`${rowIndex}-${columnIndex}`}>
              {formatValue(value)}
            </span>
          ))}
        </div>
      ))}
    </div>
  );
}

function Heatmap({
  focusedCell,
  focusedToken,
  mask,
  onFocusCell,
  onFocusToken,
  tokens,
  weights,
}) {
  const highestWeight = Math.max(...weights.flat());

  return (
    <div className="heatmap-wrap">
      <div
        className="heatmap"
        style={{ "--token-count": tokens.length }}
      >
        <span />
        {tokens.map((token, index) => (
          <span className="heatmap-axis" key={`column-${token}-${index}`}>
            {token}
          </span>
        ))}
        {weights.map((row, rowIndex) => (
          <div className="heatmap-row" key={`row-${tokens[rowIndex]}-${rowIndex}`}>
            <button
              aria-pressed={focusedToken === rowIndex}
              className="query-token"
              onClick={() => onFocusToken(rowIndex)}
              type="button"
            >
              {tokens[rowIndex]}
            </button>
            {row.map((weight, columnIndex) => {
              const intensity = weight / highestWeight;
              const isMasked = mask?.[rowIndex]?.[columnIndex];
              return (
                <button
                  aria-pressed={
                    focusedCell.query === rowIndex && focusedCell.key === columnIndex
                  }
                  className={`heat-cell${isMasked ? " masked" : ""}`}
                  key={`${rowIndex}-${columnIndex}`}
                  onClick={() => onFocusCell({ query: rowIndex, key: columnIndex })}
                  style={{ "--heat": intensity }}
                  title={
                    isMasked
                      ? `${tokens[rowIndex]} cannot attend to ${tokens[columnIndex]}`
                      : `${tokens[rowIndex]} -> ${tokens[columnIndex]}: ${formatValue(weight)}`
                  }
                >
                  {isMasked ? "mask" : formatValue(weight)}
                </button>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

function StaticHeatmap({ mask, tokens, weights }) {
  const highestWeight = Math.max(...weights.flat());

  return (
    <div className="heatmap-wrap">
      <div
        className="heatmap compact-heatmap"
        style={{ "--token-count": tokens.length }}
      >
        <span />
        {tokens.map((token, index) => (
          <span className="heatmap-axis" key={`compare-column-${token}-${index}`}>
            {token}
          </span>
        ))}
        {weights.map((row, rowIndex) => (
          <div className="heatmap-row" key={`compare-row-${rowIndex}`}>
            <span className="query-label">{tokens[rowIndex]}</span>
            {row.map((weight, columnIndex) => {
              const isMasked = mask?.[rowIndex]?.[columnIndex];
              return (
                <span
                  className={`heat-cell static${isMasked ? " masked" : ""}`}
                  key={`${rowIndex}-${columnIndex}`}
                  style={{ "--heat": weight / highestWeight }}
                >
                  {isMasked ? "mask" : formatValue(weight)}
                </span>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

function MaskComparison({ comparison }) {
  if (!comparison?.unmasked || !comparison?.masked) {
    return null;
  }

  const firstFutureWeight = comparison.unmasked.weights[0]?.[1] ?? 0;
  const maskedFutureWeight = comparison.masked.weights[0]?.[1] ?? 0;

  return (
    <section className="comparison-band" aria-label="Masked versus unmasked attention">
      <div className="comparison-head">
        <div>
          <span className="section-label">Mask comparison</span>
          <h2>Normal attention vs decoder attention</h2>
        </div>
        <p>
          Without masking, a token can look at any token. With causal masking,
          token i can only look at tokens at positions less than or equal to i.
        </p>
      </div>
      <div className="comparison-grid">
        <article>
          <header>
            <strong>Unmasked</strong>
            <span>all tokens visible</span>
          </header>
          <StaticHeatmap
            mask={comparison.unmasked.mask}
            tokens={comparison.unmasked.tokens}
            weights={comparison.unmasked.weights}
          />
        </article>
        <article>
          <header>
            <strong>Causal masked</strong>
            <span>future tokens hidden</span>
          </header>
          <StaticHeatmap
            mask={comparison.masked.mask}
            tokens={comparison.masked.tokens}
            weights={comparison.masked.weights}
          />
        </article>
      </div>
      <div className="comparison-note">
        <strong>First token future lookup</strong>
        <span>
          unmasked weight {formatValue(firstFutureWeight)} becomes{" "}
          {formatValue(maskedFutureWeight)} after masking.
        </span>
      </div>
    </section>
  );
}

function TaskModePanel({ example, onTaskModeChange, taskMode }) {
  return (
    <section className="task-band" aria-label="Learning task mode">
      <div className="task-copy">
        <span className="section-label">Learning task</span>
        <h2>{example.task_label}</h2>
        <p>{example.task_reason}</p>
        <dl>
          <div>
            <dt>Visibility rule</dt>
            <dd>{example.mask_rule}</dd>
          </div>
          <div>
            <dt>Current mask</dt>
            <dd>{example.mask ? "blocks selected token pairs" : "allows all token pairs"}</dd>
          </div>
        </dl>
      </div>
      <div className="task-picker" role="radiogroup" aria-label="Task mode">
        {taskModeOptions.map((option) => (
          <button
            aria-checked={taskMode === option.id}
            key={option.id}
            onClick={() => onTaskModeChange(option.id)}
            role="radio"
            type="button"
          >
            {option.label}
          </button>
        ))}
      </div>
    </section>
  );
}

function FormulaRail() {
  return (
    <div className="formula-rail">
      <div>
        <span>Input</span>
        <code>X = E + P</code>
      </div>
      <div>
        <span>Project</span>
        <code>Q = XWq</code>
        <code>K = XWk</code>
        <code>V = XWv</code>
      </div>
      <div>
        <span>Score</span>
        <code>S = QK^T / sqrt(dk)</code>
      </div>
      <div>
        <span>Mix</span>
        <code>A = softmax(S)</code>
        <code>Z = AV</code>
      </div>
    </div>
  );
}

function matrixShape(matrix) {
  return `${matrix.length} x ${matrix[0].length}`;
}

function VariableGlossary({ example }) {
  const variables = [
    {
      symbol: "E",
      shape: matrixShape(example.embeddings),
      name: "Token embeddings",
      detail:
        "One learned-style vector per token id. This lab looks up E from the tiny embedding table before word order is added.",
    },
    {
      symbol: "P",
      shape: matrixShape(example.position_encodings),
      name: "Position encodings",
      detail:
        "One sinusoidal vector per token position. P tells attention whether a vector came from the first, second, or later token.",
    },
    {
      symbol: "X",
      shape: matrixShape(example.attention_inputs),
      name: "Attention input",
      detail:
        "The position-aware matrix used by attention. Each row is E + P for one token in the current sentence.",
    },
    {
      symbol: "Wq, Wk, Wv",
      shape: `${matrixShape(example.query_weights)} each`,
      name: "Projection weights",
      detail:
        "Backend matrices that turn X into query, key, and value spaces. They decide what a token asks for, matches against, and passes forward.",
    },
    {
      symbol: "Q",
      shape: matrixShape(example.queries),
      name: "Queries",
      detail:
        "Q = XWq. A query row represents what the current token is searching for in other token keys.",
    },
    {
      symbol: "K",
      shape: matrixShape(example.keys),
      name: "Keys",
      detail:
        "K = XWk. A key row represents the features a token offers for score matching.",
    },
    {
      symbol: "V",
      shape: matrixShape(example.values),
      name: "Values",
      detail:
        "V = XWv. A value row is the information mixed into the final context after attention weights are known.",
    },
    {
      symbol: "dk",
      shape: `${example.attention_width}`,
      name: "Key width",
      detail:
        "The number of dimensions in each query and key vector. Scores are divided by sqrt(dk) to keep dot products controlled.",
    },
    {
      symbol: "S",
      shape: matrixShape(example.scores),
      name: "Scaled score matrix",
      detail:
        "S = QK^T / sqrt(dk). Row i scores what query token i thinks about every key token j before softmax.",
    },
    {
      symbol: "A",
      shape: matrixShape(example.weights),
      name: "Attention weights",
      detail:
        "A = softmax(S). Each row sums to 1, so it becomes a distribution over value rows for one query token.",
    },
    {
      symbol: "Z",
      shape: matrixShape(example.context),
      name: "Context output",
      detail:
        "Z = AV. Each output row is the weighted mixture of value vectors that attention sends to the next transformer step.",
    },
    {
      symbol: "M",
      shape: example.mask ? matrixShape(example.mask) : "off",
      name: "Causal mask",
      detail:
        "When enabled, M blocks score cells where a token would read the future. Those cells become zero weight after softmax.",
    },
  ];

  return (
    <div className="variable-glossary">
      <div className="glossary-head">
        <span className="section-label">Formula variables</span>
        <strong>Read the symbols</strong>
      </div>
      <div className="glossary-grid">
        {variables.map((variable) => (
          <article key={variable.symbol}>
            <header>
              <code>{variable.symbol}</code>
              <div>
                <strong>{variable.name}</strong>
                <small>shape {variable.shape}</small>
              </div>
            </header>
            <p>{variable.detail}</p>
          </article>
        ))}
      </div>
    </div>
  );
}

function LearningPipeline({ activeStep, example, onSelectStep }) {
  const steps = [
    {
      id: "tokenize",
      title: "Tokenize",
      purpose:
        "Split the sentence into small symbols and assign ids so the model can look up numeric vectors.",
      formula: "text -> tokens -> token ids",
      before: "Raw sentence",
      after: `${example.tokens.length} token ids`,
      watch:
        "Every later matrix row lines up with one token row from this step.",
      code: "tokenizer/word_tokenizer.py",
      preview: null,
    },
    {
      id: "embed",
      title: "Embed",
      purpose:
        "Turn each discrete token id into a vector. Neural math works on these vectors, not on word strings.",
      formula: "E = embedding_table[token_ids]",
      before: `${example.token_ids.length} ids`,
      after: `E shape ${matrixShape(example.embeddings)}`,
      watch:
        "Rows are tokens. Columns are embedding features chosen for this tiny lab.",
      code: "transformer_core/attention/lab.py",
      preview: { label: "Token embeddings E", matrix: example.embeddings },
    },
    {
      id: "position",
      title: "Add Position",
      purpose:
        "Add word-order information. Without this, attention sees token vectors but not where they appeared.",
      formula: "X = E + P",
      before: `E ${matrixShape(example.embeddings)} and P ${matrixShape(example.position_encodings)}`,
      after: `X shape ${matrixShape(example.attention_inputs)}`,
      watch:
        "The attention input X is the sum of token meaning and token position.",
      code: "transformer_core/attention/lab.py",
      preview: { label: "Attention inputs X", matrix: example.attention_inputs },
    },
    {
      id: "project",
      title: "Project Q/K/V",
      purpose:
        "Give each token three roles: what it asks for, what it matches with, and what information it contributes.",
      formula: "Q = XWq, K = XWk, V = XWv",
      before: `X ${matrixShape(example.attention_inputs)}`,
      after: `Q/K/V shape ${matrixShape(example.queries)}`,
      watch:
        "Projection matrices Wq, Wk, and Wv are backend weights shown below.",
      code: "transformer_core/attention/single_head.py",
      preview: { label: "Query vectors Q", matrix: example.queries },
    },
    {
      id: "score",
      title: "Score",
      purpose:
        "Compare every query with every key to estimate which tokens are relevant to one another.",
      formula: "S = QK^T / sqrt(dk)",
      before: `Q ${matrixShape(example.queries)} and K ${matrixShape(example.keys)}`,
      after: `S shape ${matrixShape(example.scores)}`,
      watch:
        "Scores are raw relevance numbers. They are not attention probabilities yet.",
      code: "transformer_core/attention/single_head.py",
      preview: { label: "Scaled scores S", matrix: example.scores },
    },
    {
      id: "mask",
      title: "Mask",
      purpose:
        "Apply the visibility rule defined by the selected learning task.",
      formula: "blocked scores -> -infinity",
      before: `S ${matrixShape(example.scores)}`,
      after: example.mask ? `${example.task_label} mask applied` : "No cells blocked",
      watch: example.mask_rule,
      code: "transformer_core/attention/single_head.py",
      preview: { label: "Masked score view", matrix: example.scores },
    },
    {
      id: "softmax",
      title: "Softmax",
      purpose:
        "Normalize each score row into weights that sum to 1 for one querying token.",
      formula: "A = softmax(S)",
      before: `S ${matrixShape(example.scores)}`,
      after: `A shape ${matrixShape(example.weights)}`,
      watch:
        "A heatmap cell is one normalized row-token to column-token attention weight.",
      code: "transformer_core/attention/single_head.py",
      preview: { label: "Attention weights A", matrix: example.weights },
    },
    {
      id: "mix",
      title: "Mix Values",
      purpose:
        "Use attention weights to blend value vectors into one context vector per token.",
      formula: "Z = AV",
      before: `A ${matrixShape(example.weights)} and V ${matrixShape(example.values)}`,
      after: `Z shape ${matrixShape(example.context)}`,
      watch:
        "The context matrix is what this attention head sends forward.",
      code: "transformer_core/attention/single_head.py",
      preview: { label: "Context output Z", matrix: example.context },
    },
  ];
  const step = steps.find((item) => item.id === activeStep) ?? steps[0];

  return (
    <section className="pipeline-band" aria-label="Guided attention pipeline">
      <div className="pipeline-head">
        <div>
          <span className="section-label">Guided pipeline</span>
          <h2>Follow one backend step at a time</h2>
        </div>
      </div>
      <div className="pipeline-tabs" role="tablist" aria-label="Attention stages">
        {steps.map((item, index) => (
          <button
            aria-selected={step.id === item.id}
            className="pipeline-tab"
            key={item.id}
            onClick={() => onSelectStep(item.id)}
            role="tab"
            type="button"
          >
            <small>{index + 1}</small>
            <span>{item.title}</span>
          </button>
        ))}
      </div>
      <div className="pipeline-stage">
        <div className="stage-copy">
          <span className="section-label">Step {steps.indexOf(step) + 1}</span>
          <h3>{step.title}</h3>
          <p>{step.purpose}</p>
          <code>{step.formula}</code>
          <dl>
            <div>
              <dt>Input</dt>
              <dd>{step.before}</dd>
            </div>
            <div>
              <dt>Output</dt>
              <dd>{step.after}</dd>
            </div>
            <div>
              <dt>Watch for</dt>
              <dd>{step.watch}</dd>
            </div>
            <div>
              <dt>Backend code</dt>
              <dd>{step.code}</dd>
            </div>
          </dl>
        </div>
        <div className="stage-preview">
          <span className="section-label">Current result</span>
          {step.preview ? (
            <>
              <strong>{step.preview.label}</strong>
              <Matrix
                columnLabels={
                  ["score", "mask", "softmax"].includes(step.id)
                    ? example.tokens
                    : step.preview.matrix[0].map((_, index) => `d${index}`)
                }
                matrix={step.preview.matrix}
                rowLabels={example.tokens}
              />
            </>
          ) : (
            <div className="token-preview">
              {example.tokens.map((token, index) => (
                <span key={`${token}-${index}`}>
                  {token}
                  <small>id {example.token_ids[index]}</small>
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

function WorkedScore({ example, selectedCell }) {
  const queryToken = example.tokens[selectedCell.query];
  const keyToken = example.tokens[selectedCell.key];
  const query = example.queries[selectedCell.query];
  const key = example.keys[selectedCell.key];
  const score = example.scores[selectedCell.query][selectedCell.key];
  const weight = example.weights[selectedCell.query][selectedCell.key];
  const isMasked = example.mask?.[selectedCell.query]?.[selectedCell.key];
  const dotProduct = query.reduce(
    (total, value, index) => total + value * key[index],
    0,
  );

  return (
    <div className="worked-score">
      <div className="worked-head">
        <span className="section-label">Selected score</span>
        <strong>
          {queryToken} to {keyToken}
        </strong>
      </div>
      <dl>
        <div>
          <dt>Query vector</dt>
          <dd>{formatVector(query)}</dd>
        </div>
        <div>
          <dt>Key vector</dt>
          <dd>{formatVector(key)}</dd>
        </div>
        <div>
          <dt>Dot product</dt>
          <dd>
            {dotTerms(query, key).join(" + ")} = {formatValue(dotProduct)}
          </dd>
        </div>
        <div>
          <dt>Scaled score</dt>
          <dd>
            {formatValue(dotProduct)} / sqrt({example.attention_width}) ={" "}
            {isMasked ? "masked" : formatValue(score)}
          </dd>
        </div>
        <div>
          <dt>Softmax weight</dt>
          <dd>{isMasked ? "0.000 after masking" : formatValue(weight)}</dd>
        </div>
      </dl>
    </div>
  );
}

function projectionCalculation(inputVector, weights, outputIndex) {
  const terms = inputVector.map((value, inputIndex) => ({
    inputLabel: `d${inputIndex}`,
    inputValue: value,
    weight: weights[inputIndex][outputIndex],
    product: value * weights[inputIndex][outputIndex],
  }));
  const total = terms.reduce((sum, term) => sum + term.product, 0);

  return { terms, total };
}

function ProjectionRole({ inputVector, outputLabels, outputVector, title, weights }) {
  return (
    <article className="projection-role">
      <header>
        <span className="section-label">{title}</span>
        <strong>{formatVector(outputVector)}</strong>
      </header>
      <div className="projection-equations">
        {outputLabels.map((label, outputIndex) => {
          const calculation = projectionCalculation(inputVector, weights, outputIndex);
          return (
            <div key={label}>
              <code>
                {label} ={" "}
                {calculation.terms
                  .map(
                    (term) =>
                      `${formatValue(term.inputValue)} x ${formatValue(term.weight)}`,
                  )
                  .join(" + ")}
              </code>
              <span>= {formatValue(calculation.total)}</span>
            </div>
          );
        })}
      </div>
    </article>
  );
}

function ProjectionInspector({ example, focusedToken, onSelectToken }) {
  const inputVector = example.attention_inputs[focusedToken];
  const token = example.tokens[focusedToken];

  return (
    <section className="projection-band" aria-label="Projection math inspector">
      <div className="projection-head">
        <div>
          <span className="section-label">Projection math</span>
          <h2>How d dimensions become Q, K, and V</h2>
        </div>
        <p>
          Select a token to see its attention input vector multiplied by Wq, Wk,
          and Wv. In real training these matrices start random and are updated
          by gradients; this lab keeps them fixed so the numbers stay repeatable.
        </p>
      </div>
      <div className="projection-token-row">
        {example.tokens.map((tokenLabel, index) => (
          <button
            aria-pressed={focusedToken === index}
            key={`${tokenLabel}-${index}`}
            onClick={() => onSelectToken(index)}
            type="button"
          >
            {tokenLabel}
          </button>
        ))}
      </div>
      <div className="projection-input">
        <span className="section-label">Selected input</span>
        <strong>
          X for {token}: {formatVector(inputVector)}
        </strong>
        <small>
          This vector uses columns d0, d1, d2. Each projection matrix maps those
          input dimensions into a new two-dimensional role space.
        </small>
      </div>
      <div className="projection-grid">
        <ProjectionRole
          inputVector={inputVector}
          outputLabels={["q0", "q1"]}
          outputVector={example.queries[focusedToken]}
          title="Q = XWq"
          weights={example.query_weights}
        />
        <ProjectionRole
          inputVector={inputVector}
          outputLabels={["k0", "k1"]}
          outputVector={example.keys[focusedToken]}
          title="K = XWk"
          weights={example.key_weights}
        />
        <ProjectionRole
          inputVector={inputVector}
          outputLabels={["v0", "v1"]}
          outputVector={example.values[focusedToken]}
          title="V = XWv"
          weights={example.value_weights}
        />
      </div>
    </section>
  );
}

function LossCurve({ metrics }) {
  if (!metrics?.length) {
    return (
      <div className="empty-chart">
        <span>No training metrics yet</span>
      </div>
    );
  }

  const points = metrics.filter((metric, index) => index % 2 === 0 || index === metrics.length - 1);
  const maxLoss = Math.max(...points.map((metric) => metric.loss));
  const minLoss = Math.min(...points.map((metric) => metric.loss));
  const range = Math.max(maxLoss - minLoss, 0.0001);
  const polylinePoints = points
    .map((metric, index) => {
      const x = points.length === 1 ? 0 : (index / (points.length - 1)) * 100;
      const y = 92 - ((metric.loss - minLoss) / range) * 78;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <div className="loss-chart">
      <svg viewBox="0 0 100 100" role="img" aria-label="Training loss curve">
        <polyline points={polylinePoints} />
      </svg>
      <div className="chart-axis">
        <span>{formatMetric(maxLoss)}</span>
        <span>{formatMetric(minLoss)}</span>
      </div>
      <div className="chart-caption">
        <span>step {points[0].step}</span>
        <span>step {points[points.length - 1].step}</span>
      </div>
    </div>
  );
}

function buildEpochRows(summary) {
  const stepsPerEpoch = Math.max(1, Math.floor(summary.metrics.length / summary.epochs));

  return Array.from({ length: summary.epochs }, (_, epochIndex) => {
    const start = epochIndex * stepsPerEpoch;
    const end = Math.min(start + stepsPerEpoch, summary.metrics.length);
    const metrics = summary.metrics.slice(start, end);
    const lastMetric = metrics[metrics.length - 1] ?? summary.metrics[summary.metrics.length - 1];
    const averageLoss =
      metrics.reduce((total, metric) => total + metric.loss, 0) / Math.max(metrics.length, 1);
    const averageAccuracy =
      metrics.reduce((total, metric) => total + metric.accuracy, 0) /
      Math.max(metrics.length, 1);

    return {
      accuracy: averageAccuracy,
      averageLoss,
      epoch: epochIndex + 1,
      lastLoss: lastMetric.loss,
      steps: metrics.length,
    };
  });
}

function TrainingEpochLog({ summary }) {
  const epochRows = buildEpochRows(summary);
  const visibleRows = epochRows.length > 14
    ? [...epochRows.slice(0, 6), ...epochRows.slice(-8)]
    : epochRows;
  const skippedCount = Math.max(epochRows.length - visibleRows.length, 0);

  return (
    <article className="training-panel epoch-log-panel">
      <div className="training-panel-head">
        <span className="section-label">Epoch steps</span>
        <strong>{summary.epochs} epochs</strong>
      </div>
      <div className="epoch-log">
        {visibleRows.map((row, index) => (
          <div className="epoch-row" key={`epoch-${row.epoch}`}>
            {skippedCount > 0 && index === 6 ? (
              <div className="epoch-skip">
                <span>{skippedCount} epochs hidden</span>
              </div>
            ) : null}
            <div className="epoch-row-head">
              <strong>
                Epoch {row.epoch}/{summary.epochs}
              </strong>
              <span>
                {row.steps}/{row.steps} steps
              </span>
            </div>
            <div className="epoch-progress" aria-hidden="true">
              <span style={{ "--progress": "100%" }} />
            </div>
            <div className="epoch-metrics">
              <span>loss: {formatMetric(row.lastLoss)}</span>
              <span>avg_loss: {formatMetric(row.averageLoss)}</span>
              <span>acc: {formatMetric(row.accuracy)}</span>
            </div>
          </div>
        ))}
      </div>
    </article>
  );
}

function NumberField({ label, min, max, onChange, step = "1", value }) {
  return (
    <label className="number-field">
      <span>{label}</span>
      <input
        max={max}
        min={min}
        onChange={(event) => onChange(event.target.value)}
        step={step}
        type="number"
        value={value}
      />
    </label>
  );
}

function TrainingDashboard() {
  const [config, setConfig] = useState({
    context_size: 1,
    embedding_learning_rate: 0.01,
    epochs: 40,
    learning_rate: 0.08,
    train_embeddings: true,
    train_transformer: false,
  });
  const [example, setExample] = useState(null);
  const [trainingRun, setTrainingRun] = useState(null);
  const [trainingError, setTrainingError] = useState("");
  const [isTraining, setIsTraining] = useState(false);

  useEffect(() => {
    let isMounted = true;

    async function loadTrainingData() {
      try {
        const response = await fetch("/api/training/example");
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail ?? `API returned ${response.status}`);
        }

        let latestPayload = null;
        const latestResponse = await fetch("/api/training/latest");
        if (latestResponse.ok) {
          latestPayload = await latestResponse.json();
        }

        if (isMounted) {
          setExample(payload);
          setConfig(payload.default_request);
          if (latestPayload) {
            setTrainingRun(latestPayload);
          }
        }
      } catch (error) {
        if (isMounted) {
          setTrainingError(error.message);
        }
      }
    }

    loadTrainingData();

    return () => {
      isMounted = false;
    };
  }, []);

  async function runTraining(event) {
    event.preventDefault();
    setIsTraining(true);
    setTrainingError("");

    try {
      const response = await fetch("/api/training/run", {
        body: JSON.stringify({
          context_size: Number(config.context_size),
          embedding_learning_rate: Number(config.embedding_learning_rate),
          epochs: Number(config.epochs),
          learning_rate: Number(config.learning_rate),
          train_embeddings: Boolean(config.train_embeddings),
          train_transformer: Boolean(config.train_transformer),
        }),
        headers: { "Content-Type": "application/json" },
        method: "POST",
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail ?? `API returned ${response.status}`);
      }
      setTrainingRun(payload);
    } catch (error) {
      setTrainingError(error.message);
    } finally {
      setIsTraining(false);
    }
  }

  const summary = trainingRun?.summary;
  const vocabularyEntries = summary
    ? Object.entries(summary.vocabulary).sort((left, right) => left[1] - right[1])
    : example
      ? Object.entries(example.vocabulary).sort((left, right) => left[1] - right[1])
      : [];
  const tokenLabels = vocabularyEntries.map(([token]) => token);

  return (
    <section className="training-dashboard" aria-label="Tiny GPT training dashboard">
      <div className="training-hero">
        <div>
          <span className="section-label">Training dashboard</span>
          <h2>Tiny next-token training</h2>
          <p>
            This run trains the final vocabulary projection and, optionally,
            token embeddings while the transformer stack stays frozen.
          </p>
        </div>
        <form className="training-controls" onSubmit={runTraining}>
          <NumberField
            label="Epochs"
            max="200"
            min="1"
            onChange={(epochs) => setConfig({ ...config, epochs })}
            value={config.epochs}
          />
          <NumberField
            label="Learning rate"
            max="1"
            min="0.001"
            onChange={(learning_rate) => setConfig({ ...config, learning_rate })}
            step="0.001"
            value={config.learning_rate}
          />
          <NumberField
            label="Embedding rate"
            max="1"
            min="0.001"
            onChange={(embedding_learning_rate) =>
              setConfig({ ...config, embedding_learning_rate })
            }
            step="0.001"
            value={config.embedding_learning_rate}
          />
          <NumberField
            label="Context"
            max="5"
            min="1"
            onChange={(context_size) => setConfig({ ...config, context_size })}
            value={config.context_size}
          />
          <label className="train-toggle">
            <input
              checked={config.train_embeddings}
              onChange={(event) =>
                setConfig({ ...config, train_embeddings: event.target.checked })
              }
              type="checkbox"
            />
            <span>Train embeddings</span>
          </label>
          <label className="train-toggle">
            <input
              checked={config.train_transformer}
              onChange={(event) =>
                setConfig({ ...config, train_transformer: event.target.checked })
              }
              type="checkbox"
            />
            <span>Train transformer</span>
          </label>
          <button disabled={isTraining} type="submit">
            {isTraining ? "Training" : "Run training"}
          </button>
        </form>
      </div>

      {trainingError ? <span className="run-error">{trainingError}</span> : null}

      {summary ? (
        <>
          <div className="training-summary-grid">
            <article>
              <span>Initial loss</span>
              <strong>{formatMetric(summary.initial_loss)}</strong>
            </article>
            <article>
              <span>Final loss</span>
              <strong>{formatMetric(summary.final_loss)}</strong>
            </article>
            <article>
              <span>Before</span>
              <strong>{summary.initial_predictions.join(", ")}</strong>
            </article>
            <article>
              <span>After</span>
              <strong>{summary.final_predictions.join(", ")}</strong>
            </article>
            <article>
              <span>Target</span>
              <strong>{summary.target_token_ids.join(", ")}</strong>
            </article>
          </div>

          <div className="training-main-grid">
            <article className="training-panel">
              <div className="training-panel-head">
                <span className="section-label">Loss curve</span>
                <strong>{summary.metrics.length} steps</strong>
              </div>
              <LossCurve metrics={summary.metrics} />
            </article>

            <article className="training-panel">
              <div className="training-panel-head">
                <span className="section-label">Run setup</span>
                <strong>
                  {summary.train_transformer
                    ? "Full transformer"
                    : summary.train_embeddings
                      ? "Projection + embeddings"
                      : "Projection only"}
                </strong>
              </div>
              <dl className="run-detail-list">
                <div>
                  <dt>Examples</dt>
                  <dd>{summary.example_count}</dd>
                </div>
                <div>
                  <dt>Epochs</dt>
                  <dd>{summary.epochs}</dd>
                </div>
                <div>
                  <dt>Context size</dt>
                  <dd>{summary.context_size}</dd>
                </div>
                <div>
                  <dt>Run path</dt>
                  <dd>{trainingRun.run_path}</dd>
                </div>
              </dl>
            </article>
          </div>

          <TrainingEpochLog summary={summary} />

          <div className="training-tables">
            <article className="training-panel">
              <div className="training-panel-head">
                <span className="section-label">Token embeddings</span>
                <strong>E</strong>
              </div>
              <Matrix
                columnLabels={summary.token_embeddings[0].map((_, index) => `d${index}`)}
                matrix={summary.token_embeddings}
                rowLabels={tokenLabels}
              />
            </article>

            <article className="training-panel">
              <div className="training-panel-head">
                <span className="section-label">Vocabulary projection</span>
                <strong>Wout</strong>
              </div>
              <Matrix
                columnLabels={tokenLabels}
                matrix={summary.vocabulary_weights}
                rowLabels={summary.vocabulary_weights.map((_, index) => `h${index}`)}
              />
            </article>
          </div>
        </>
      ) : (
        <div className="training-empty">
          <span>{example ? "Run training to create metrics" : "Loading training config"}</span>
        </div>
      )}
    </section>
  );
}

export function App() {
  const [activeView, setActiveView] = useState("attention");
  const [activeTensor, setActiveTensor] = useState("embeddings");
  const [attentionExample, setAttentionExample] = useState(null);
  const [attentionComparison, setAttentionComparison] = useState(null);
  const [loadError, setLoadError] = useState("");
  const [runError, setRunError] = useState("");
  const [sentence, setSentence] = useState("");
  const [taskMode, setTaskMode] = useState("classification");
  const [isRunning, setIsRunning] = useState(false);
  const [focusedToken, setFocusedToken] = useState(1);
  const [selectedCell, setSelectedCell] = useState({ query: 1, key: 1 });
  const [activeLearningStep, setActiveLearningStep] = useState("tokenize");

  useEffect(() => {
    let isMounted = true;

    async function loadAttentionExample() {
      try {
        const response = await fetch("/api/attention/example");
        if (!response.ok) {
          throw new Error(`API returned ${response.status}`);
        }

        const payload = await response.json();
        const compareResponse = await fetch("/api/attention/compare", {
          body: JSON.stringify({ text: payload.text }),
          headers: { "Content-Type": "application/json" },
          method: "POST",
        });
        const comparisonPayload = await compareResponse.json();
        if (!compareResponse.ok) {
          throw new Error(
            comparisonPayload.detail ?? `API returned ${compareResponse.status}`,
          );
        }

        if (isMounted) {
          setAttentionExample(payload);
          setAttentionComparison(comparisonPayload);
          setSentence(payload.text);
          setTaskMode(payload.task_mode);
          setFocusedToken(Math.min(1, payload.tokens.length - 1));
          setSelectedCell({
            query: Math.min(1, payload.tokens.length - 1),
            key: Math.min(1, payload.tokens.length - 1),
          });
        }
      } catch (error) {
        if (isMounted) {
          setLoadError(error.message);
        }
      }
    }

    loadAttentionExample();

    return () => {
      isMounted = false;
    };
  }, []);

  if (loadError) {
    return (
      <main className="studio-shell status-shell">
        <section className="status-panel">
          <p>MiniGPT Studio</p>
          <h1>Attention API unavailable</h1>
          <span>{loadError}</span>
        </section>
      </main>
    );
  }

  if (!attentionExample) {
    return (
      <main className="studio-shell status-shell">
        <section className="status-panel">
          <p>MiniGPT Studio</p>
          <h1>Loading attention tensors</h1>
        </section>
      </main>
    );
  }

  const focusedWeights = attentionExample.weights[focusedToken];
  const strongestWeight = Math.max(...focusedWeights);
  const strongestTarget = focusedWeights.indexOf(strongestWeight);
  const tensor = attentionExample[activeTensor];

  async function runAttention(event) {
    event.preventDefault();
    await runAttentionRequest(taskMode);
  }

  async function runAttentionRequest(nextTaskMode) {
    setIsRunning(true);
    setRunError("");

    try {
      const response = await fetch("/api/attention/run", {
        body: JSON.stringify({ task_mode: nextTaskMode, text: sentence }),
        headers: { "Content-Type": "application/json" },
        method: "POST",
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail ?? `API returned ${response.status}`);
      }

      const compareResponse = await fetch("/api/attention/compare", {
        body: JSON.stringify({ text: sentence }),
        headers: { "Content-Type": "application/json" },
        method: "POST",
      });
      const comparisonPayload = await compareResponse.json();
      if (!compareResponse.ok) {
        throw new Error(comparisonPayload.detail ?? `API returned ${compareResponse.status}`);
      }

      setAttentionExample(payload);
      setTaskMode(payload.task_mode);
      setAttentionComparison(comparisonPayload);
      const nextFocusedToken = Math.min(focusedToken, payload.tokens.length - 1);
      setFocusedToken(nextFocusedToken);
      setSelectedCell({
        query: nextFocusedToken,
        key: Math.min(selectedCell.key, payload.tokens.length - 1),
      });
    } catch (error) {
      setRunError(error.message);
    } finally {
      setIsRunning(false);
    }
  }

  function changeTaskMode(nextTaskMode) {
    setTaskMode(nextTaskMode);
    runAttentionRequest(nextTaskMode);
  }

  return (
    <main className="studio-shell">
      <header className="studio-head">
        <p>MiniGPT Studio</p>
        <h1>{activeView === "attention" ? "Single-head attention lab" : "Training studio"}</h1>
        <nav className="studio-nav" aria-label="Studio sections">
          <button
            aria-pressed={activeView === "attention"}
            onClick={() => setActiveView("attention")}
            type="button"
          >
            Attention Lab
          </button>
          <button
            aria-pressed={activeView === "training"}
            onClick={() => setActiveView("training")}
            type="button"
          >
            Training Dashboard
          </button>
        </nav>
        {activeView === "training" ? null : (
        <form className="sentence-form" onSubmit={runAttention}>
          <input
            aria-label="Attention sentence"
            maxLength="120"
            onChange={(event) => setSentence(event.target.value)}
            type="text"
            value={sentence}
          />
          <button disabled={isRunning} type="submit">
            {isRunning ? "Running" : "Run"}
          </button>
        </form>
        )}
        {runError ? <span className="run-error">{runError}</span> : null}
      </header>

      {activeView === "training" ? (
        <TrainingDashboard />
      ) : (
        <>

      <section className="sentence-band" aria-label="Tokenized sentence">
        <div>
          <span className="section-label">Sentence</span>
          <h2>{attentionExample.text}</h2>
        </div>
        <div className="token-strip">
          {attentionExample.tokens.map((token, index) => (
            <button
              aria-pressed={focusedToken === index}
              className="token"
              key={`${token}-${index}`}
              onClick={() => setFocusedToken(index)}
              type="button"
            >
              <span>{token}</span>
              <small>id {attentionExample.token_ids[index]}</small>
            </button>
          ))}
        </div>
      </section>

      <TaskModePanel
        example={attentionExample}
        onTaskModeChange={changeTaskMode}
        taskMode={taskMode}
      />

      <LearningPipeline
        activeStep={activeLearningStep}
        example={attentionExample}
        onSelectStep={setActiveLearningStep}
      />

      <MaskComparison comparison={attentionComparison} />

      <section className="math-band" aria-label="Attention math">
        <div className="math-overview">
          <span className="section-label">Backend math</span>
          <FormulaRail />
          <VariableGlossary example={attentionExample} />
        </div>
        <WorkedScore example={attentionExample} selectedCell={selectedCell} />
      </section>

      <section className="visual-grid">
        <div className="heatmap-panel">
          <div className="panel-title">
            <span className="section-label">Attention weights</span>
            <p>
              <strong>{attentionExample.tokens[focusedToken]}</strong> attends
              most to <strong>{attentionExample.tokens[strongestTarget]}</strong>.
            </p>
          </div>
          <Heatmap
            focusedCell={selectedCell}
            focusedToken={focusedToken}
            mask={attentionExample.mask}
            onFocusCell={(cell) => {
              setFocusedToken(cell.query);
              setSelectedCell(cell);
            }}
            onFocusToken={setFocusedToken}
            tokens={attentionExample.tokens}
            weights={attentionExample.weights}
          />
        </div>

        <div className="score-panel">
          <span className="section-label">Scaled scores</span>
          <Matrix
            columnLabels={attentionExample.tokens}
            matrix={attentionExample.scores}
            rowLabels={attentionExample.tokens}
          />
        </div>
      </section>

      <section className="tensor-band" aria-label="Attention tensors">
        <div className="tensor-tabs" role="tablist" aria-label="Tensor view">
          {tensorViews.map((view) => (
            <button
              aria-selected={activeTensor === view.key}
              className="tensor-tab"
              key={view.key}
              onClick={() => setActiveTensor(view.key)}
              role="tab"
              type="button"
            >
              {view.label}
            </button>
          ))}
        </div>
        <Matrix
          columnLabels={tensor[0].map((_, index) => `d${index}`)}
          matrix={tensor}
          rowLabels={attentionExample.tokens}
        />
      </section>

      <ProjectionInspector
        example={attentionExample}
        focusedToken={focusedToken}
        onSelectToken={(tokenIndex) => {
          setFocusedToken(tokenIndex);
          setSelectedCell({
            query: tokenIndex,
            key: Math.min(selectedCell.key, attentionExample.tokens.length - 1),
          });
        }}
      />

      <section className="weight-band" aria-label="Projection weights">
        <div>
          <span className="section-label">Projection weights</span>
          <h2>Backend matrices</h2>
        </div>
        <div className="weight-grid">
          <div>
            <strong>Wq</strong>
            <Matrix
              columnLabels={attentionExample.query_weights[0].map((_, index) => `q${index}`)}
              matrix={attentionExample.query_weights}
              rowLabels={attentionExample.attention_inputs[0].map((_, index) => `x${index}`)}
            />
          </div>
          <div>
            <strong>Wk</strong>
            <Matrix
              columnLabels={attentionExample.key_weights[0].map((_, index) => `k${index}`)}
              matrix={attentionExample.key_weights}
              rowLabels={attentionExample.attention_inputs[0].map((_, index) => `x${index}`)}
            />
          </div>
          <div>
            <strong>Wv</strong>
            <Matrix
              columnLabels={attentionExample.value_weights[0].map((_, index) => `v${index}`)}
              matrix={attentionExample.value_weights}
              rowLabels={attentionExample.attention_inputs[0].map((_, index) => `x${index}`)}
            />
          </div>
        </div>
      </section>
        </>
      )}
    </main>
  );
}
