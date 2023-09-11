function initAllCharts() {
  createPopulation();
  createElderlyPopulationChart();
  createPopulationPyramid();
  createElderlyDependencyChart();
  setTimeout(function() {
    window.dispatchEvent(new Event('resize'));
  }, 0);
  
}
