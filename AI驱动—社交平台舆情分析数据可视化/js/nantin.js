const mychart = echarts.init(document.getElementById("nantin"));
option = {
    title: {
        text: '微博内容情感占比（百分比）',
        subtext: '整件事件中人们发布的微博的情感类型',
      left: 'left',
      top: 0,
      textStyle: {
          fontSize: 15,
          color: '#fff',
      }
  },
yAxis: {
type: 'category',
data: ['积极', '消极']
},
xAxis: {
type: 'value'
},
series: [
{
data: [{
    value: 52,
    itemStyle: {
      color: '#66cc66'
    }
    }, {
        value: 48,
            itemStyle: {
                color: '#a91354'
            }
        }],
type: 'bar',

}
]
};
mychart.setOption(option);
window.addEventListener('resize', function () { mychart.resize(); });