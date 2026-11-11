const fs = require('fs');
const path = require('path');

const outDir = path.join(__dirname, 'canva_tfg_deck_v2');
const assetSrc = path.join(__dirname, 'canva_tfg_deck', 'assets');
const assetDst = path.join(outDir, 'assets');
fs.mkdirSync(assetDst, { recursive: true });
if (fs.existsSync(assetSrc)) {
  for (const file of fs.readdirSync(assetSrc)) {
    const src = path.join(assetSrc, file);
    const dst = path.join(assetDst, file);
    if (fs.statSync(src).isFile()) fs.copyFileSync(src, dst);
  }
}

const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const img = (file, alt = '') => `<img class="figure" src="assets/${esc(file)}" alt="${esc(alt)}">`;
const bullets = arr => `<ul>${arr.map(x => `<li>${x}</li>`).join('')}</ul>`;
const cards = arr => `<div class="cards ${arr.length === 2 ? 'two' : arr.length === 4 ? 'four' : ''}">${arr.map((c, i) => `<div class="card accent-${i % 4}"><strong>${c[0]}</strong><p>${c[1]}</p></div>`).join('')}</div>`;
const flow = arr => `<div class="flow">${arr.map((x, i) => `<div class="step accent-${i % 4}">${x}</div>`).join('')}</div>`;
const score = rows => `<div class="score">${rows.map(r => `<div><span>${r[0]}</span><strong>${r[1]}</strong></div>`).join('')}</div>`;
const chartBars = rows => `<div class="bar-chart">${rows.map(([label, value, max, cls = '']) => `<div class="bar-row ${cls}"><span>${label}</span><div><i style="width:${Math.max(4, value / max * 100).toFixed(1)}%"></i></div><b>${value}</b></div>`).join('')}</div>`;

const slides = [
  {
    kicker: 'Title',
    title: 'Training Neural Networks with Evolution Strategies',
    subtitle: 'A Michigan-Style Evolution Strategy for Hidden-Layer Construction',
    body: `<div class="title-meta">Pau Prat Moreno<br>Bachelor’s Degree in Artificial Intelligence<br>Universitat Politècnica de Catalunya — BarcelonaTech<br>Thesis supervisor: Luis Antonio Belanche Muñoz</div>`,
    notes: 'Introduce the thesis in one sentence: instead of evolving complete neural networks, I evolve individual hidden neurons and let the selected population become a hidden layer.'
  },
  {
    kicker: 'Research Question',
    title: 'Can We Evolve a Hidden Layer Neuron by Neuron?',
    body: `<div class="split"><div>${bullets(['Standard neuroevolution: one individual = one complete neural network', 'Proposed method: one individual = one hidden neuron', 'Research question: can this representation be useful and efficient?'])}</div><div class="compare"><div>Complete<br>network</div><span>vs</span><div>Single<br>neuron</div></div></div>`,
    notes: 'Frame the thesis as a study of representation. The key difference is not merely a different optimiser but a different meaning for an evolutionary individual.'
  },
  {
    kicker: 'Training Context',
    title: 'Beyond Gradient-Based Training',
    body: cards([
      ['Backpropagation', 'Jointly updates all parameters through gradients.'],
      ['Evolution Strategies', 'Search through mutation and selection.'],
      ['Gradient-free search', 'A continuous optimisation alternative.']
    ]),
    notes: 'Clarify that this work does not argue against backpropagation. Backpropagation remains the standard method. The goal is to investigate a different learning dynamic for compact neural models.'
  },
  {
    kicker: 'Test Bed',
    title: 'A Controlled and Demanding Test Bed',
    body: `<div class="split"><div>${bullets(['Fixed-length numerical feature vectors', 'Strong classical baselines remain competitive', 'Suitable for compact, interpretable experiments', 'Focus: binary classification, regression, and multiclass classification'])}</div><div class="mini-net"><span>tabular data</span><b>→</b><span>hidden layer</span><b>→</b><span>output</span></div></div>`,
    notes: 'Explain that tabular data are useful because the model must add value beyond linear methods, random features, and strong classical learners.'
  },
  {
    kicker: 'Motivation',
    title: 'Why Evolve Individual Neurons?',
    body: cards([
      ['Compact object', 'Weights, bias, and mutation strategy.'],
      ['Continuous search', 'Evolution Strategies naturally mutate real-valued parameters.'],
      ['Cooperative layer', 'A population can become one hidden representation.'],
      ['Potential efficiency', 'Cheaper than evaluating many complete networks.']
    ]),
    notes: 'Emphasise the conceptual appeal: each neuron is a natural unit for an Evolution Strategy, but cooperation between neurons creates the main challenge.'
  },
  {
    kicker: 'Objectives',
    title: 'Objectives',
    body: cards([
      ['1', 'Build a Michigan-style ES-MLP prototype.'],
      ['2', 'Design neuron-level credit assignment.'],
      ['3', 'Fit task-appropriate output layers.'],
      ['4', 'Test whether the evolved representation adds value.'],
      ['5', 'Compare Michigan and Pittsburgh representations.']
    ]),
    notes: 'State that the objectives are empirical and architectural. The thesis does not claim universal dominance or a new convergence theorem.'
  },
  {
    kicker: 'Roadmap',
    title: 'Roadmap',
    body: flow(['Representation and background', 'Proposed method', 'Experimental protocol', 'Results', 'Conclusions']),
    notes: 'Briefly preview the presentation. Keep this slide fast.'
  },
  {
    kicker: 'Neural Model',
    title: 'The Neural Model: A Learned Nonlinear Feature Map',
    body: `<div class="equation">h<sub>j</sub>(x) = tanh(w<sub>j</sub><sup>T</sup>x + b<sub>j</sub>)</div>${bullets(['Each hidden neuron transforms the input into one nonlinear feature', 'The output layer combines the resulting hidden features'])}<div class="mini-net wide"><span>input features</span><b>→</b><span>tanh neurons</span><b>→</b><span>output model</span></div>`,
    notes: 'Explain that the hidden layer can be understood as a learned feature map. This viewpoint is crucial because the evolutionary process searches for hidden features.'
  },
  {
    kicker: 'Feature View',
    title: 'A Hidden Neuron Is a Candidate Feature',
    body: `<div class="matrix"><div></div><div></div><div class="selected"></div><div></div><div class="selected"></div><div></div><div></div><div class="selected"></div><div></div><div></div><div></div><div class="selected"></div></div>${bullets(['Each neuron produces one activation column over the dataset', 'The hidden layer is an activation matrix Φ', 'Selecting neurons is equivalent to selecting nonlinear transformed features'])}`,
    notes: 'Make the bridge to feature selection. A neuron does not predict independently; it contributes one candidate feature to the final predictive model.'
  },
  {
    kicker: 'Representation',
    title: 'Two Evolutionary Representations',
    body: `<div class="split"><div><h3>Pittsburgh</h3>${bullets(['One individual = one complete neural network', 'Fitness is direct: evaluate the model'])}</div><div><h3>Michigan</h3>${bullets(['One individual = one hidden neuron', 'Population collectively forms one hidden layer'])}</div></div>${img('illustration_pittsburgh_vs_michigan.png', 'Pittsburgh versus Michigan conceptual figure')}`,
    notes: 'Explain that Pittsburgh is simpler for fitness assignment because every individual is a complete solution. Michigan is more granular but must solve cooperative credit assignment.'
  },
  {
    kicker: 'Challenge',
    title: 'Credit Assignment: Which Neurons Deserve to Survive?',
    body: `<div class="venn"><span>relevance</span><span>diversity</span><span>cooperation</span></div>${bullets(['Relevant to the target', 'Different from selected neurons', 'Complementary to the current hidden representation'])}`,
    notes: 'Explain redundancy. A neuron can look strong in isolation but add no new information. Conversely, a moderately predictive neuron may be valuable if it explains residual information.'
  },
  {
    kicker: 'Evolution Strategies',
    title: 'Evolution Strategies at Neuron Level',
    body: `${flow(['Parent neurons', 'Self-adaptive mutation', 'Offspring neurons', 'Parent + offspring pool', 'Selection'])}<div class="equation small-eq">Σ<sub>i</sub> = L<sub>i</sub>L<sub>i</sub><sup>T</sup></div>`,
    notes: 'Explain that every neuron has its own mutation behaviour. Some neurons can explore locally, while others can make broader moves in parameter space.'
  },
  {
    kicker: 'Hypothesis',
    title: 'Hypothesis',
    body: `<div class="statement">Evolving individual hidden neurons can produce useful nonlinear representations and may offer a favourable accuracy-time trade-off compared with evolving complete neural networks.</div>`,
    notes: 'State this as the central claim tested by the rest of the presentation.'
  },
  {
    kicker: 'Architecture',
    title: 'Prototype Overview',
    body: `${flow(['Data preprocessing', 'Michigan ES', 'Neuron selection', 'Evolved hidden layer', 'Task-specific output layer'])}${img('prototype_architecture.png', 'Prototype architecture diagram')}`,
    notes: 'Explain the separation between representation learning and output fitting. Evolution searches for hidden neurons; the output layer is fitted after hidden-layer selection.'
  },
  {
    kicker: 'Individual Encoding',
    title: 'One Individual = One Hidden Neuron',
    body: `<div class="neuron-card"><strong>Neuron individual</strong><span>input weights</span><span>bias</span><span>mutation matrix</span><span>tanh activation</span></div>${bullets(['Input weights', 'Bias', 'Self-adaptive mutation matrix', 'tanh activation function'])}`,
    notes: 'Stress that an individual is deliberately small. This reduces the unit of mutation and selection from a whole network to one feature-producing component.'
  },
  {
    kicker: 'Training Cycle',
    title: 'Training Cycle',
    body: flow(['Retain μ parents', 'Generate λ offspring', 'Form candidate pool', 'Compute activations', 'Select μ neurons', 'Fit output layer', 'Keep best checkpoint']),
    notes: 'Walk through one generation. Mention that the selected population becomes the hidden layer for that generation.'
  },
  {
    kicker: 'Selection',
    title: 'Four Credit-Assignment Strategies',
    body: cards([
      ['Marginal R² + diversity', 'Individually predictive neurons.'],
      ['OMP', 'Neurons explaining residual information.'],
      ['CFS', 'Target relevance with low redundancy.'],
      ['CFS-AUC', 'Ranking relevance with monotonic redundancy.']
    ]),
    notes: 'Explain that these are four answers to the same question: how to choose neurons that are jointly useful, rather than merely individually correlated with the target.'
  },
  {
    kicker: 'CFS',
    title: 'CFS: Relevance Without Redundancy',
    body: `<div class="split"><div>${bullets(['Select neurons strongly related to the target', 'Penalise similarity among selected neurons', 'Encourage compact and cooperative hidden representations'])}</div><div>${img('illustration_credit_assignment.png', 'Redundant cluster versus selected subset')}</div></div>`,
    notes: 'Explain why CFS is conceptually aligned with the Michigan representation. The goal is not a ranking of independent neurons but a subset that works together.'
  },
  {
    kicker: 'Output Layers',
    title: 'One Hidden Representation, Three Output Models',
    body: cards([
      ['Binary classification', 'Logistic regression + cross-entropy.'],
      ['Regression', 'Ridge regression + Generalised Cross-Validation.'],
      ['Multiclass classification', 'Softmax + cross-entropy.']
    ]) + img('task_specific_output_layer.png', 'Task-specific output layers'),
    notes: 'Explain why output models are matched to the task. Ridge/GCV is appropriate for continuous targets; logistic and softmax outputs provide probabilities for classification.'
  },
  {
    kicker: 'Design Choice',
    title: 'Separating Representation Learning from Output Fitting',
    body: `<div class="split"><div><h3>Standard MLP</h3>${bullets(['Joint optimisation of hidden and output weights'])}</div><div><h3>Proposed ES-MLP</h3>${bullets(['Evolve nonlinear hidden features', 'Fit the final predictor afterwards'])}</div></div>`,
    notes: 'Clarify that the architecture separates nonlinear feature discovery from supervised prediction. This makes the method easier to inspect and compare.'
  },
  {
    kicker: 'Trade-Offs',
    title: 'Design Trade-Offs',
    body: `<div class="split"><div><h3>Benefits</h3>${bullets(['Fine-grained evolutionary search', 'Explicit diversity among hidden features', 'Flexible output models'])}</div><div><h3>Risks</h3>${bullets(['Difficult credit assignment', 'Redundancy and overfitting', 'Limited scalability'])}</div></div>`,
    notes: 'Present the method honestly. It is not automatically superior; it only works if selection can build a useful cooperative hidden layer.'
  },
  {
    kicker: 'Evaluation',
    title: 'What Was Evaluated?',
    body: bullets(['Which neuron-selection strategy works best?', 'Does evolution improve over linear and random-feature models?', 'Can the approach support regression and multiclass tasks?', 'Can the selected hidden layer be compressed?', 'Does Michigan outperform Pittsburgh evolution?']),
    notes: 'Explain that the evaluation is a sequence of diagnostic experiments, not a single benchmark leaderboard.'
  },
  {
    kicker: 'Datasets',
    title: 'Benchmark Tasks',
    body: cards([
      ['Binary classification', 'Breast-W · Ionosphere · Sonar'],
      ['Regression', 'California Housing · Friedman1 · Diabetes'],
      ['Multiclass', 'Digits · Gaussian Quantiles · Spiral']
    ]),
    notes: 'Briefly state that the binary benchmarks are the main representation study, while regression and multiclass experiments test transferability of the architecture.'
  },
  {
    kicker: 'Protocol',
    title: 'Reproducibility and Fair Evaluation',
    body: bullets(['Train / validation / test separation', 'Scaling and imputation fitted on training data only', 'Validation used for model selection and checkpointing', 'Test set reserved for final reporting', 'Multiple seeds for stochastic experiments']),
    notes: 'Explain that validation control is especially important because evolutionary training evaluates many generations and could otherwise overfit design decisions to test data.'
  },
  {
    kicker: 'Metrics',
    title: 'How Performance Was Measured',
    body: cards([
      ['Classification', 'Accuracy, ROC-AUC, cross-entropy, Brier score and ECE.'],
      ['Regression', 'RMSE, MAE, R².'],
      ['Efficiency', 'Fit time and per-generation workload.']
    ]),
    notes: 'State that accuracy alone is insufficient. The thesis also evaluates ranking quality, calibration, regression quality, generalisation gaps, and runtime.'
  },
  {
    kicker: 'Result 1',
    title: 'Credit Assignment Matters',
    body: `<div class="headline">CFS was the strongest overall default strategy.</div>${bullets(['Highest aggregate validation accuracy', 'Highest aggregate test accuracy: 0.881', 'Highest aggregate ROC-AUC: 0.930', 'Best overall Brier score and calibration behaviour among the tested strategies'])}${img('strategy_cost_summary.png', 'Selection strategy summary')}`,
    notes: 'Explain that selection strategy materially changes the final hidden representation. CFS was selected as the default because it offered the strongest aggregate balance, not because it won every dataset or every metric.'
  },
  {
    kicker: 'Result 1b',
    title: 'Performance Must Justify Cost',
    body: chartBars([['R²', 0.513, 45], ['OMP', 1.012, 45], ['CFS', 1.234, 45, 'highlight'], ['CFS-AUC', 45.0, 45]]) + `<div class="headline">CFS-AUC was much more expensive without a consistent predictive advantage.</div>`,
    notes: 'Explain that CFS-AUC was theoretically attractive but computationally disproportionate. CFS gave a much better practical performance-cost balance.'
  },
  {
    kicker: 'Result 2',
    title: 'Does Evolution Add Value?',
    body: `<div class="metrics-grid">${score([['Linear Logistic CE', 'Acc 0.876 · AUC 0.901'], ['Random tanh Logistic CE', 'Acc 0.857 · AUC 0.895'], ['ES-CFS Logistic CE', 'Acc 0.881 · AUC 0.930']])}</div><div class="headline">The gain is not explained merely by adding nonlinear features.</div>`,
    notes: 'Make the key inference: evolved features beat both linear features and random nonlinear features. Therefore, neuron evolution and cooperative selection add measurable value.'
  },
  {
    kicker: 'Result 3',
    title: 'Comparison with Standard Tabular Models',
    body: chartBars([['RBF SVM', .918, .918], ['Random Forest', .911, .918], ['HistGradientBoosting', .908, .918], ['Logistic Regression', .895, .918], ['ES-CFS Logistic CE', .881, .918, 'highlight'], ['Linear Logistic CE', .876, .918], ['Gradient MLP', .859, .918], ['Random tanh Logistic CE', .857, .918]]),
    notes: 'Be transparent: ES-CFS does not outperform the strongest classical tabular baselines. However, it improves over linear logistic, random nonlinear features, and the tested gradient-trained MLP.'
  },
  {
    kicker: 'Result 4',
    title: 'Regression with Ridge/GCV',
    body: `<div class="split"><div>${chartBars([['Linear Ridge', .593, .769], ['Random tanh Ridge', .704, .769], ['ES-selected Ridge', .769, .769, 'highlight']])}</div><div>${bullets(['Friedman1: ES-selected representation improves substantially over linear and random-feature Ridge', 'Diabetes: ES-selected Ridge also achieves the strongest representation-based result'])}</div></div><div class="headline">The evolved hidden layer improves the Ridge/GCV output model across all tested regression datasets.</div>`,
    notes: 'Explain that the best neuron-selection strategy changed across regression datasets, so CFS was not assumed to be universally best. The main result is that the evolved representation added value to the Ridge/GCV output layer.'
  },
  {
    kicker: 'Result 5',
    title: 'The Hidden Layer Can Be Compressed',
    body: `<div class="compress"><span>50 neurons</span><b>→</b><span>≈18 active neurons</span></div>${bullets(['Evolution selects a useful hidden layer', 'L1-logistic output filtering removes unnecessary neurons', 'Approximately 18 active neurons retained from an initial 50', 'Sparse output filtering slightly improves average test accuracy and ROC-AUC'])}`,
    notes: 'Explain that evolutionary selection is a first structural filter, while L1 logistic regression provides a second supervised filter. This suggests the final representation contains redundancy that can be removed without harming performance.'
  },
  {
    kicker: 'Result 6',
    title: 'Multiclass Softmax Models',
    body: bullets(['Digits: ES-CFS Softmax is competitive, but Gradient MLP remains stronger', 'Gaussian Quantiles: ES-CFS Softmax reaches near-perfect nonlinear classification', 'Spiral: ES-CFS Softmax reaches near-perfect nonlinear classification']) + `<div class="headline">The evolved representation is especially effective when linear softmax is structurally insufficient.</div>`,
    notes: 'Explain that the method shows its strongest benefit on clearly nonlinear multiclass tasks. On Digits, the result is more mixed and should be interpreted cautiously.'
  },
  {
    kicker: 'Result 7',
    title: 'Michigan vs Full-Network Pittsburgh Evolution',
    body: `<div class="split"><div><h3>Michigan ES-CFS</h3>${score([['Test accuracy', '0.882'], ['Test ROC-AUC', '0.935'], ['Validation-test gap', '0.042'], ['Fit time', '0.444 s']])}</div><div><h3>Pittsburgh ES</h3>${score([['Test accuracy', '0.818'], ['Test ROC-AUC', '0.864'], ['Validation-test gap', '0.180'], ['Fit time', '4.873 s']])}</div></div>`,
    notes: 'This is the key result. Pittsburgh achieved almost perfect validation accuracy but generalised worse. Michigan had higher test accuracy, higher AUC, much lower overfitting gap, and around eleven times lower training time.'
  },
  {
    kicker: 'Result 8',
    title: 'Controlling for the Output Solver',
    body: `<div class="split"><div><h3>Michigan ES-CFS</h3>${score([['New neurons / generation', '250'], ['Output fits / generation', '1'], ['Test accuracy', '0.882'], ['Test AUC', '0.935'], ['Fit time', '0.444 s']])}</div><div><h3>Pittsburgh matched-budget</h3>${score([['New neurons / generation', '250'], ['Output fits / generation', '10'], ['Test accuracy', '0.816'], ['Test AUC', '0.840'], ['Fit time', '1.474 s']])}</div></div>`,
    notes: 'Explain why this control matters. It removes the difference in the number of newly generated hidden neurons and uses the same logistic output solver. The result still favours Michigan.'
  },
  {
    kicker: 'Interpretation',
    title: 'What Did the Thesis Demonstrate?',
    body: cards([
      ['1', 'A population of evolved neurons can form a useful hidden representation.'],
      ['2', 'Credit assignment is decisive; CFS is the best default in the tested binary setting.'],
      ['3', 'The evolved representation improves over linear and random-feature alternatives.'],
      ['4', 'Michigan provides a better tested generalisation-cost trade-off than Pittsburgh evolution.']
    ]),
    notes: 'State the main contribution precisely. The thesis supports the Michigan idea within the tested setting; it does not show universal superiority over all machine-learning methods.'
  },
  {
    kicker: 'Conclusions',
    title: 'Conclusions and Future Work',
    body: `<div class="split"><div><h3>Limitations</h3>${bullets(['Single-hidden-layer architecture', 'Limited benchmark scope', 'Specific Pittsburgh baselines', 'Calibration can still be improved'])}</div><div><h3>Future work</h3>${bullets(['Deeper Michigan architectures', 'Adaptive hidden-layer size', 'Improved calibration and regularisation', 'Larger benchmark suites', 'Neuron stability and interpretability analysis'])}</div></div><div class="headline">Evolving neurons instead of networks can be a practical way to construct nonlinear hidden representations.</div>`,
    notes: 'End with a balanced conclusion. The contribution is not replacing backpropagation or tree ensembles, but demonstrating that population representation is a meaningful architectural design choice in neuroevolution.'
  }
];

const html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TFG-Pau_Prat_Moreno_v2</title>
  <style>
    :root {
      --blue:#1f6fbf; --teal:#009e73; --gold:#e69f00; --ink:#111827;
      --muted:#526071; --line:#d9e2ec; --soft:#f5f8fb; --white:#ffffff;
    }
    *{box-sizing:border-box}
    html,body{margin:0;background:#ffffff;color:var(--ink);font-family:Inter,Aptos,Arial,sans-serif}
    .slide{width:1600px;height:900px;position:relative;overflow:hidden;padding:70px 90px 62px;background:#ffffff;page-break-after:always;border-bottom:1px solid #edf1f5}
    .slide::before{content:"";position:absolute;left:0;top:0;width:10px;height:100%;background:var(--blue)}
    .accent-line{width:72px;height:7px;background:var(--teal);margin-bottom:22px}
    .kicker{font-size:21px;text-transform:uppercase;letter-spacing:.08em;color:var(--blue);font-weight:800;margin-bottom:16px}
    h1{font-size:62px;line-height:1.05;letter-spacing:0;margin:0;max-width:1220px}
    .subtitle{font-size:31px;line-height:1.24;color:var(--muted);margin-top:22px;max-width:1160px}
    .title-meta{font-size:27px;line-height:1.36;color:var(--ink);margin-top:58px}
    .content{margin-top:34px}
    h3{font-size:31px;color:var(--blue);margin:0 0 18px}
    p,li{font-size:30px;line-height:1.28}
    ul{margin:0;padding-left:36px}
    li{margin:14px 0}
    .footer{position:absolute;left:90px;bottom:28px;font-size:18px;color:#8190a0}
    .num{position:absolute;right:90px;bottom:28px;font-size:18px;color:#8190a0}
    .cards{display:grid;grid-template-columns:repeat(3,1fr);gap:24px}
    .cards.two{grid-template-columns:repeat(2,1fr)}
    .cards.four{grid-template-columns:repeat(4,1fr)}
    .card{border:2px solid var(--line);background:#ffffff;border-radius:8px;padding:26px 26px 24px;min-height:172px;box-shadow:0 6px 18px rgba(15,23,42,.05)}
    .card::before{content:"";display:block;width:54px;height:6px;background:var(--blue);margin-bottom:18px}
    .accent-1::before,.step.accent-1{background:var(--teal)}
    .accent-2::before,.step.accent-2{background:var(--gold)}
    .accent-3::before,.step.accent-3{background:#64748b}
    .card strong{font-size:30px;line-height:1.1;color:var(--ink);display:block;margin-bottom:12px}
    .card p{font-size:24px;color:var(--muted);margin:0;line-height:1.25}
    .split{display:grid;grid-template-columns:1fr 1fr;gap:48px;align-items:center}
    .compare{display:grid;grid-template-columns:1fr auto 1fr;gap:22px;align-items:center;text-align:center}
    .compare div,.mini-net span,.compress span{border:2px solid var(--line);border-radius:8px;padding:38px 20px;background:#fff;font-size:31px;font-weight:800}
    .compare span,.mini-net b,.compress b{color:var(--teal);font-size:30px}
    .mini-net{display:flex;gap:18px;align-items:center;justify-content:center}
    .mini-net.wide{margin-top:34px}
    .mini-net span{font-size:24px;padding:24px 30px}
    .flow{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px;margin-top:10px}
    .step{min-height:132px;border-radius:8px;padding:22px;display:flex;align-items:center;justify-content:center;text-align:center;color:#fff;font-size:25px;font-weight:800;line-height:1.15}
    .equation{background:var(--soft);border:2px solid var(--line);border-radius:8px;padding:24px;text-align:center;font-family:Cambria,Georgia,serif;font-size:43px;margin-bottom:28px}
    .small-eq{max-width:420px;margin:40px auto 0}
    .matrix{display:grid;grid-template-columns:repeat(12,1fr);gap:8px;height:230px;margin-bottom:34px}
    .matrix div{background:#edf2f7;border:1px solid var(--line);border-radius:6px}
    .matrix .selected{background:#d6f4ee;border-color:var(--teal)}
    .figure{width:100%;max-height:360px;object-fit:contain;margin-top:20px;border:1px solid var(--line);border-radius:8px;background:#fff}
    .venn{height:310px;position:relative;margin-bottom:24px}
    .venn span{position:absolute;width:300px;height:300px;border:5px solid;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:26px;font-weight:800;background:rgba(255,255,255,.72)}
    .venn span:nth-child(1){left:345px;top:0;border-color:var(--blue)}
    .venn span:nth-child(2){left:555px;top:0;border-color:var(--teal)}
    .venn span:nth-child(3){left:450px;top:135px;border-color:var(--gold)}
    .statement{font-size:51px;line-height:1.18;font-weight:800;max-width:1220px;border-left:11px solid var(--teal);padding-left:32px;margin-top:80px}
    .neuron-card{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;border:2px solid var(--line);border-radius:8px;padding:30px;background:#fff;margin-bottom:34px}
    .neuron-card strong{grid-column:1/-1;font-size:34px;color:var(--blue)}
    .neuron-card span{border:1px solid var(--line);padding:22px;border-radius:8px;text-align:center;font-size:25px;background:var(--soft)}
    .headline{font-size:31px;font-weight:800;color:var(--ink);border-left:9px solid var(--teal);padding:16px 0 16px 24px;margin-top:26px;background:#f7fbfa}
    .bar-chart{display:grid;gap:11px}
    .bar-row{display:grid;grid-template-columns:330px 1fr 92px;gap:18px;align-items:center;font-size:23px}
    .bar-row div{height:32px;background:#eef3f7;border-radius:4px;overflow:hidden}
    .bar-row i{display:block;height:100%;background:#9fb5c9}
    .bar-row.highlight i{background:var(--teal)}
    .bar-row b{text-align:right}
    .score{display:grid;gap:12px}
    .score div{display:flex;justify-content:space-between;gap:20px;border-bottom:2px solid var(--line);padding:13px 0;font-size:25px}
    .score strong{font-size:27px;color:var(--blue)}
    .compress{display:flex;align-items:center;justify-content:center;gap:30px;margin:20px 0 40px}
    .speaker-notes{display:none}
  </style>
</head>
<body>
${slides.map((s, i) => `<section class="slide">
  <div class="accent-line"></div>
  <div class="kicker">${esc(s.kicker)}</div>
  <h1>${esc(s.title)}</h1>
  ${s.subtitle ? `<div class="subtitle">${esc(s.subtitle)}</div>` : ''}
  <div class="content">${s.body}</div>
  <aside class="speaker-notes">${esc(s.notes)}</aside>
  <div class="footer">TFG · Michigan-style ES-MLP</div>
  <div class="num">${i + 1} / ${slides.length}</div>
</section>`).join('\n')}
</body>
</html>`;

fs.writeFileSync(path.join(outDir, 'index.html'), html);
fs.writeFileSync(path.join(outDir, 'speaker-notes.txt'), slides.map((s, i) => `Slide ${i + 1}: ${s.title}\n${s.notes}\n`).join('\n'));
console.log(path.join(outDir, 'index.html'));
