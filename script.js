const imageInput = document.getElementById('imageInput');
const jsonInput = document.getElementById('jsonInput');
const trainBtn = document.getElementById('trainBtn');
const stopBtn = document.getElementById('stopBtn');
const downloadBtn = document.getElementById('downloadBtn');
const statusText = document.getElementById('statusText');
const epochText = document.getElementById('epochText');
const origSizeText = document.getElementById('origSizeText');
const compSizeText = document.getElementById('compSizeText');

const origCanvas = document.getElementById('originalCanvas');
const origCtx = origCanvas.getContext('2d');
const outCanvas = document.getElementById('outputCanvas');

const TRAIN_SIZE = 64;   // 学習解像度 (64x64)
const RENDER_SIZE = 256; // 出力解像度 (256x256)

let model = null;
let coordsTensor = null;
let colorsTensor = null;
let renderCoordsTensor = null;
let isTraining = false;
let epoch = 0;

// Custom Sinレイヤー（SIRENアーキテクチャ）
class SineLayer extends tf.layers.Layer {
  constructor(config) {
    super(config);
    this.omega = config.omega || 30.0;
  }
  call(inputs) {
    return tf.tidy(() => {
      const x = Array.isArray(inputs) ? inputs[0] : inputs;
      return tf.sin(tf.mul(x, this.omega));
    });
  }
  static get className() {
    return 'SineLayer';
  }
}
tf.serialization.registerClass(SineLayer);

// 斜め方向の周波数成分を含む2Dフーリエ特徴量マッピング
const DIR_VECTORS = [
  [1, 0], [0, 1],
  [2, 0], [0, 2], [1.5, 1.5], [-1.5, 1.5],
  [4, 0], [0, 4], [3, 3], [-3, 3],
  [8, 0], [0, 8], [6, 6], [-6, 6],
  [14, 0], [0, 14]
];

function encodeCoords(x, y) {
  const features = [x, y];
  for (const [fx, fy] of DIR_VECTORS) {
    const phase = Math.PI * (fx * x + fy * y);
    features.push(Math.sin(phase));
    features.push(Math.cos(phase));
  }
  return features; // 合計34次元
}

// 高精度・高速収束モデルの構築
function createINRModel() {
  const inputDim = 2 + DIR_VECTORS.length * 2;
  const m = tf.sequential();

  m.add(tf.layers.dense({ inputShape: [inputDim], units: 64, kernelInitializer: 'glorotUniform' }));
  m.add(new SineLayer({ omega: 1.5 }));

  m.add(tf.layers.dense({ units: 64, kernelInitializer: 'glorotUniform' }));
  m.add(new SineLayer({ omega: 1.5 }));

  m.add(tf.layers.dense({ units: 64, kernelInitializer: 'glorotUniform' }));
  m.add(new SineLayer({ omega: 1.5 }));

  // 出力: RGB (0〜1)
  m.add(tf.layers.dense({ units: 3, activation: 'sigmoid' }));

  m.compile({
    optimizer: tf.train.adam(0.008),
    loss: 'meanSquaredError'
  });
  return m;
}

// 描画用の高解像度座標 (256x256) を事前に準備
function prepareRenderCoords() {
  if (renderCoordsTensor) return renderCoordsTensor;
  const coords = [];
  for (let y = 0; y < RENDER_SIZE; y++) {
    for (let x = 0; x < RENDER_SIZE; x++) {
      const nx = (x / (RENDER_SIZE - 1)) * 2 - 1;
      const ny = (y / (RENDER_SIZE - 1)) * 2 - 1;
      coords.push(encodeCoords(nx, ny));
    }
  }
  renderCoordsTensor = tf.tensor2d(coords);
  return renderCoordsTensor;
}

// 画像選択時の処理（圧縮用）
imageInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;

  isTraining = false;
  const reader = new FileReader();
  reader.onload = (event) => {
    const img = new Image();
    img.onload = () => {
      origCtx.clearRect(0, 0, TRAIN_SIZE, TRAIN_SIZE);
      origCtx.drawImage(img, 0, 0, TRAIN_SIZE, TRAIN_SIZE);
      const imgData = origCtx.getImageData(0, 0, TRAIN_SIZE, TRAIN_SIZE).data;

      const coords = [];
      const colors = [];

      for (let y = 0; y < TRAIN_SIZE; y++) {
        for (let x = 0; x < TRAIN_SIZE; x++) {
          const idx = (y * TRAIN_SIZE + x) * 4;
          const nx = (x / (TRAIN_SIZE - 1)) * 2 - 1;
          const ny = (y / (TRAIN_SIZE - 1)) * 2 - 1;
          coords.push(encodeCoords(nx, ny));

          colors.push([
            imgData[idx] / 255,
            imgData[idx + 1] / 255,
            imgData[idx + 2] / 255
          ]);
        }
      }

      if (coordsTensor) coordsTensor.dispose();
      if (colorsTensor) colorsTensor.dispose();

      coordsTensor = tf.tensor2d(coords);
      colorsTensor = tf.tensor2d(colors);
      prepareRenderCoords();

      if (model) model.dispose();
      model = createINRModel();
      epoch = 0;

      const rawBytes = TRAIN_SIZE * TRAIN_SIZE * 3;
      const weightCount = model.countParams();
      const weightBytesF16 = weightCount * 2;
      
      origSizeText.textContent = `元サイズ: ${(rawBytes / 1024).toFixed(2)} KB (64×64)`;
      compSizeText.textContent = `AI重みサイズ: 約 ${(weightBytesF16 / 1024).toFixed(2)} KB (${weightCount} params)`;

      statusText.textContent = '・準備完了 「圧縮開始」を押してください';
      epochText.textContent = 'Epoch: 0';
      trainBtn.disabled = false;
      stopBtn.disabled = true;
      downloadBtn.disabled = true;
    };
    img.src = event.target.result;
  };
  reader.readAsDataURL(file);
});

// 【新機能】JSONファイル選択時の処理（デコード・復元用）
jsonInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;

  isTraining = false;
  statusText.textContent = '・JSONデータを読み込んで復元中...';

  const reader = new FileReader();
  reader.onload = async (event) => {
    try {
      const payload = JSON.parse(event.target.result);
      
      if (model) model.dispose();
      model = createINRModel();
      prepareRenderCoords();

      // JSONから取得した重みデータをテンソルに変換してモデルにセット
      const weights = payload.weights.map(wData => {
        return tf.tensor(wData.data, wData.shape);
      });
      model.setWeights(weights);
      weights.forEach(w => w.dispose());

      // キャンバスをクリアして復元画像を描画
      origCtx.clearRect(0, 0, TRAIN_SIZE, TRAIN_SIZE);
      origCtx.fillStyle = '#000';
      origCtx.fillRect(0, 0, TRAIN_SIZE, TRAIN_SIZE);

      await renderOutputFast();

      const totalParams = model.countParams();
      origSizeText.textContent = `元サイズ: 復元モード`;
      compSizeText.textContent = `AI重みサイズ: ${(file.size / 1024).toFixed(2)} KB (${totalParams} params)`;
      statusText.textContent = '・復元完了！ JSONから画像を正常に再現しました';
      epochText.textContent = 'Epoch: Loaded';
      
      trainBtn.disabled = true;
      stopBtn.disabled = true;
      downloadBtn.disabled = false;
    } catch (err) {
      console.error(err);
      statusText.textContent = '・エラー: 不正なJSONファイルです';
    }
  };
  reader.readAsText(file);
});

// 高速化された学習ループ（学習率の自動減衰付き）
async function trainLoop() {
  const totalPixels = TRAIN_SIZE * TRAIN_SIZE;
  const MAX_EPOCHS = 2000;

  while (isTraining && epoch < MAX_EPOCHS) {
    const decayedLr = Math.max(0.0005, 0.008 * Math.pow(0.998, epoch));
    model.optimizer.learningRate = decayedLr;

    const history = await model.fit(coordsTensor, colorsTensor, {
      batchSize: totalPixels,
      epochs: 15,
      shuffle: false,
      verbose: 0
    });
    
    epoch += 15;
    const loss = history.history.loss[history.history.loss.length - 1];
    const psnr = (-10 * Math.log10(Math.max(loss, 1e-10))).toFixed(1);
    
    epochText.textContent = `Epoch: ${epoch}`;
    statusText.textContent = `・学習中... Loss: ${loss.toFixed(5)} (画質指標: ${psnr} dB)`;

    await renderOutputFast();
    await tf.nextFrame();
  }

  if (epoch >= MAX_EPOCHS) {
    isTraining = false;
    statusText.textContent = `・圧縮完了！ (最終Epoch: ${epoch})`;
    trainBtn.disabled = false;
    stopBtn.disabled = true;
    downloadBtn.disabled = false;
  }
}

// GPUから直接出力Canvasへ高速描画
async function renderOutputFast() {
  const imgTensor = tf.tidy(() => {
    const preds = model.predict(renderCoordsTensor);
    return preds.reshape([RENDER_SIZE, RENDER_SIZE, 3]);
  });
  await tf.browser.toPixels(imgTensor, outCanvas);
  imgTensor.dispose();
}

// ボタンイベント
trainBtn.addEventListener('click', () => {
  if (!model || isTraining) return;
  isTraining = true;
  trainBtn.disabled = true;
  stopBtn.disabled = false;
  downloadBtn.disabled = true;
  statusText.textContent = '・ニューラルネットワークを学習中...';
  trainLoop();
});

stopBtn.addEventListener('click', async () => {
  if (!isTraining) return;
  isTraining = false;
  statusText.textContent = `・一時停止中 (Epoch: ${epoch})`;
  trainBtn.disabled = false;
  stopBtn.disabled = true;
  downloadBtn.disabled = false;
  await renderOutputFast();
});

// 圧縮データ(.json)のダウンロード
downloadBtn.addEventListener('click', async () => {
  if (!model) return;
  statusText.textContent = '・圧縮JSONデータを生成中...';

  const weights = model.getWeights();
  const weightData = await Promise.all(weights.map(async (w) => {
    const raw = await w.data();
    const compact = Array.from(raw, v => Math.round(v * 10000) / 10000);
    return {
      shape: w.shape,
      data: compact
    };
  }));

  const payload = {
    format: 'INR-SIREN-v1',
    weights: weightData
  };

  const blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `inr_compressed_${Date.now()}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);

  statusText.textContent = `・保存完了！ (JSONサイズ: ${(blob.size / 1024).toFixed(2)} KB)`;
});