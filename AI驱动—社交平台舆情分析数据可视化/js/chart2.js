const chart = echarts.init(document.getElementById('chart2'));
// prettier-ignore
const dataAxis = ['持续时间','强度','成片','等局','市域','降水','层云','降雨量','落区','降水区','黑手','灾性','稳定性','单点','旱区','锋面','积雨云','降水强度','桥隧','速度慢'];
// prettier-ignore
const data = [0.8991494178771973,0.896638810634613,
0.8906767964363098,0.8899631500244141,0.8844988346099854,0.8798621296882629,0.8726263642311096,0.8696848750114441,0.8693847060203552,0.8678573369979858,0.865398108959198,0.8652946949005127,0.8651469349861145,0.8650116324424744,0.8644891977310181,0.8639407753944397,0.8618673086166382,0.8600186109542847,0.8589925765991211,0.8580343127250671];
const yMax = 1;
const dataShadow = [];
for (let i = 0; i < data.length; i++) {
dataShadow.push(yMax);
}
const option = {
    title: {
        text: '与“洪水”相关度最大的20个词汇',
        subtext: '滚动滚轮拖动柱状图查看',
        textStyle: {
            fontSize: 15,
            color: '#fff',
        }
    },
    xAxis: {
      data: dataAxis,
      axisLabel: {
        inside: false,
        color: '#999'
      },
      axisTick: {
        show: false
      },
      axisLine: {
        show: false
      },
      z: 10
    },
    yAxis: {
      min: 0.85,
      axisLine: {
        show: true
      },
      axisTick: {
        show: false
      },
      axisLabel: {
        color: '#999'
      }
    },
    dataZoom: [
      {
            type: 'inside',
            xAxisIndex: 0
      }
    ],
    series: [
      {
        type: 'bar',
        showBackground: true,
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: '#83bff6' },
            { offset: 0.5, color: '#188df0' },
            { offset: 1, color: '#188df0' }
          ])
        },
        emphasis: {
          itemStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: '#2378f7' },
              { offset: 0.7, color: '#2378f7' },
              { offset: 1, color: '#83bff6' }
            ])
          }
        },
        data: data
      }
    ]
  };
// Enable data zoom when user click bar.
const zoomSize = 6;
chart.on('click', function (params) {
  console.log(dataAxis[Math.max(params.dataIndex - zoomSize / 2, 0)]); 
    chart.dispatchAction({
    type: 'dataZoom',
    startValue: dataAxis[Math.max(params.dataIndex - zoomSize / 2, 0)],
    endValue:
      dataAxis[Math.min(params.dataIndex + zoomSize / 2, data.length - 1)]
  });
});
chart.setOption(option);
window.addEventListener('resize', function () { chart.resize(); });
