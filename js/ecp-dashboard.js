window.ECP_VIEW_SERIES = window.ECP_VIEW_SERIES || [
  { label: "Mar 03", value: 126 },
  { label: "Mar 04", value: 148 },
  { label: "Mar 05", value: 162 },
  { label: "Mar 06", value: 171 },
  { label: "Mar 07", value: 158 },
  { label: "Mar 08", value: 196 },
  { label: "Mar 09", value: 214 },
  { label: "Mar 10", value: 238 },
  { label: "Mar 11", value: 221 },
  { label: "Mar 12", value: 267 }
];

(function() {
  function formatNumber(value) {
    return value.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }

  function buildChartSvg(series, width, height) {
    var padding = { top: 20, right: 18, bottom: 42, left: 20 };
    var innerWidth = width - padding.left - padding.right;
    var innerHeight = height - padding.top - padding.bottom;
    var maxValue = Math.max.apply(null, series.map(function(point) { return point.value; }));
    var minValue = 0;
    var valueRange = Math.max(maxValue - minValue, 1);
    var stepX = series.length > 1 ? innerWidth / (series.length - 1) : innerWidth;

    function xFor(index) {
      return padding.left + stepX * index;
    }

    function yFor(value) {
      return padding.top + innerHeight - ((value - minValue) / valueRange) * innerHeight;
    }

    var linePath = "";
    var areaPath = "";
    var points = [];
    var gridLines = [];
    var axisLabels = [];
    var pointDots = [];
    var i;

    for (i = 0; i < 4; i += 1) {
      var fraction = i / 3;
      var y = padding.top + innerHeight * fraction;
      var tickValue = Math.round(maxValue - (maxValue - minValue) * fraction);
      gridLines.push('<line x1="' + padding.left + '" y1="' + y + '" x2="' + (width - padding.right) + '" y2="' + y + '"></line>');
      axisLabels.push('<text x="' + padding.left + '" y="' + (y - 8) + '">' + tickValue + '</text>');
    }

    for (i = 0; i < series.length; i += 1) {
      var px = xFor(i);
      var py = yFor(series[i].value);
      points.push(px + "," + py);
      pointDots.push(
        '<g class="chart-point">' +
        '<circle cx="' + px + '" cy="' + py + '" r="5"></circle>' +
        '<title>' + series[i].label + ": " + series[i].value + "</title>" +
        "</g>"
      );
      axisLabels.push('<text class="x-label" x="' + px + '" y="' + (height - 10) + '">' + series[i].label + "</text>");
    }

    linePath = "M " + points.join(" L ");
    areaPath = linePath + " L " + xFor(series.length - 1) + " " + (height - padding.bottom) + " L " + xFor(0) + " " + (height - padding.bottom) + " Z";

    return '' +
      '<svg viewBox="0 0 ' + width + " " + height + '" role="img" aria-labelledby="ecp-chart-title">' +
      '<defs>' +
      '<linearGradient id="ecpChartFill" x1="0" x2="0" y1="0" y2="1">' +
      '<stop offset="0%" stop-color="rgba(137, 96, 229, 0.42)"></stop>' +
      '<stop offset="100%" stop-color="rgba(137, 96, 229, 0.04)"></stop>' +
      "</linearGradient>" +
      '<linearGradient id="ecpChartStroke" x1="0" x2="1" y1="0" y2="0">' +
      '<stop offset="0%" stop-color="#7f58f2"></stop>' +
      '<stop offset="100%" stop-color="#5bc9ff"></stop>' +
      "</linearGradient>" +
      "</defs>" +
      '<title id="ecp-chart-title">Page view trend</title>' +
      '<g class="chart-grid">' + gridLines.join("") + "</g>" +
      '<g class="chart-axis">' + axisLabels.join("") + "</g>" +
      '<path class="chart-area" d="' + areaPath + '"></path>' +
      '<path class="chart-line" d="' + linePath + '"></path>' +
      '<g class="chart-points">' + pointDots.join("") + "</g>" +
      "</svg>";
  }

  function renderEcpDashboard() {
    var series = window.ECP_VIEW_SERIES || [];
    var chartContainer = document.getElementById("ecp-view-chart");
    var totalEl = document.getElementById("ecp-view-total");
    var peakEl = document.getElementById("ecp-view-peak");
    var peakLabelEl = document.getElementById("ecp-view-peak-label");
    var windowEl = document.getElementById("ecp-view-window");
    var rangeEl = document.getElementById("ecp-view-range");

    if (!chartContainer || !series.length) {
      return;
    }

    var total = 0;
    var peak = series[0];

    series.forEach(function(point) {
      total += point.value;
      if (point.value > peak.value) {
        peak = point;
      }
    });

    if (totalEl) {
      totalEl.textContent = formatNumber(total);
    }
    if (peakEl) {
      peakEl.textContent = formatNumber(peak.value);
    }
    if (peakLabelEl) {
      peakLabelEl.textContent = peak.label;
    }
    if (windowEl) {
      windowEl.textContent = series.length + " pts";
    }
    if (rangeEl) {
      rangeEl.textContent = series[0].label + " to " + series[series.length - 1].label;
    }

    var width = Math.max(chartContainer.clientWidth, 320);
    var height = 300;
    chartContainer.innerHTML = buildChartSvg(series, width, height);
  }

  window.addEventListener("load", renderEcpDashboard);
  window.addEventListener("resize", renderEcpDashboard);
})();
