(() => {
  const svg = document.getElementById('loss-chart');
  const status = document.getElementById('chart-status');
  if (!svg || !status) return;

  const NS = 'http://www.w3.org/2000/svg';
  const width = 920;
  const height = 390;
  const margin = { top: 28, right: 28, bottom: 54, left: 68 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const add = (name, attrs = {}, text = '') => {
    const node = document.createElementNS(NS, name);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    if (text) node.textContent = text;
    svg.appendChild(node);
    return node;
  };

  const renderFrame = () => {
    svg.replaceChildren();
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-labelledby', 'loss-chart-title loss-chart-description');
    add('title', { id: 'loss-chart-title' }, 'Training and validation loss by epoch');
    add('desc', { id: 'loss-chart-description' }, 'Pending chart with no training observations yet.');
    add('rect', { x: margin.left, y: margin.top, width: plotWidth, height: plotHeight, fill: '#fffffb', stroke: '#d8d4cc' });
    for (let i = 0; i <= 5; i += 1) {
      const y = margin.top + (plotHeight * i) / 5;
      add('line', { x1: margin.left, x2: width - margin.right, y1: y, y2: y, stroke: '#ebe8e1' });
    }
    for (let i = 0; i <= 6; i += 1) {
      const x = margin.left + (plotWidth * i) / 6;
      add('line', { x1: x, x2: x, y1: margin.top, y2: height - margin.bottom, stroke: '#f2f0ea' });
      add('text', { x, y: height - 27, 'text-anchor': 'middle', fill: '#66635f', 'font-family': 'system-ui, sans-serif', 'font-size': 11 }, String(i * 5));
    }
    add('text', { x: margin.left + plotWidth / 2, y: height - 6, 'text-anchor': 'middle', fill: '#66635f', 'font-family': 'system-ui, sans-serif', 'font-size': 12 }, 'Epoch');
    add('text', { x: 17, y: margin.top + plotHeight / 2, transform: `rotate(-90 17 ${margin.top + plotHeight / 2})`, 'text-anchor': 'middle', fill: '#66635f', 'font-family': 'system-ui, sans-serif', 'font-size': 12 }, 'Huber loss');
  };

  const renderPending = (message) => {
    add('line', { x1: margin.left + 45, x2: width - margin.right - 45, y1: margin.top + plotHeight / 2, y2: margin.top + plotHeight / 2, stroke: '#d8d4cc', 'stroke-dasharray': '7 7' });
    add('text', { x: margin.left + plotWidth / 2, y: margin.top + plotHeight / 2 - 12, 'text-anchor': 'middle', fill: '#6b5c92', 'font-family': 'system-ui, sans-serif', 'font-size': 18, 'font-weight': 650 }, 'Training pending');
    add('text', { x: margin.left + plotWidth / 2, y: margin.top + plotHeight / 2 + 19, 'text-anchor': 'middle', fill: '#66635f', 'font-family': 'system-ui, sans-serif', 'font-size': 12 }, message);
  };

  const renderRuns = (runs) => {
    const points = runs.flatMap((run) => run.epochs || []);
    if (!points.length) {
      renderPending('No epoch observations are available yet.');
      return;
    }
    const maxEpoch = Math.max(...points.map((point) => Number(point.epoch)), 30);
    const losses = points.flatMap((point) => [Number(point.train_loss), Number(point.validation_loss)]).filter(Number.isFinite);
    const minLoss = Math.min(...losses);
    const maxLoss = Math.max(...losses);
    const padding = Math.max((maxLoss - minLoss) * 0.12, 0.01);
    const low = Math.max(0, minLoss - padding);
    const high = maxLoss + padding;
    const x = (epoch) => margin.left + (Number(epoch) / maxEpoch) * plotWidth;
    const y = (loss) => margin.top + ((high - Number(loss)) / (high - low)) * plotHeight;

    for (let i = 0; i <= 5; i += 1) {
      const value = high - ((high - low) * i) / 5;
      const yy = margin.top + (plotHeight * i) / 5;
      add('text', { x: margin.left - 10, y: yy + 4, 'text-anchor': 'end', fill: '#66635f', 'font-family': 'system-ui, sans-serif', 'font-size': 11 }, value.toFixed(3));
    }
    const colors = { train_loss: '#006d77', validation_loss: '#9b4d00' };
    runs.forEach((run) => {
      ['train_loss', 'validation_loss'].forEach((field) => {
        const valid = (run.epochs || []).filter((point) => Number.isFinite(Number(point[field])));
        if (!valid.length) return;
        const coordinates = valid.map((point) => `${x(point.epoch)},${y(point[field])}`).join(' ');
        add('polyline', { points: coordinates, fill: 'none', stroke: colors[field], 'stroke-width': 2.5, 'stroke-linejoin': 'round', 'stroke-linecap': 'round', opacity: .92 });
        valid.forEach((point) => add('circle', { cx: x(point.epoch), cy: y(point[field]), r: 2.5, fill: colors[field] }));
      });
    });
    svg.querySelector('desc').textContent = `Loss curves for ${runs.length} training run${runs.length === 1 ? '' : 's'}.`;
  };

  renderFrame();
  fetch('assets/training-history.json', { cache: 'no-store' })
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then((data) => {
      status.textContent = data.status === 'pending' ? 'Awaiting first epoch' : `Updated ${data.updated_at}`;
      if (Array.isArray(data.runs) && data.runs.some((run) => (run.epochs || []).length)) renderRuns(data.runs);
      else renderPending(data.message || 'No epoch observations are available yet.');
    })
    .catch(() => {
      status.textContent = 'Training history unavailable';
      renderPending('No epoch observations are available yet.');
    });
})();
