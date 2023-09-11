     // 初始化ECharts实例
     var myChart = echarts.init(document.getElementById('move'));

     // 初始化数据
     var dates = []; // 日期
var totalWeiboCounts = []; // 累积微博数量
var label0Counts = []; // 情感标签值为0的数量
var label1Counts = []; // 情感标签值为1的数量

     // 使用fetch API加载本地CSV文件
     fetch('/datasets/emotion.csv')
         .then(response => response.text())
         .then(data => {
             // 解析CSV数据
             var lines = data.split('\n');
             var currentDate = ''; // 当前日期
             var totalCount = 0; // 当前日期的总微博数量

             var label0Count = 0; // 当前日期的情感标签值为0的数量
             var label1Count = 0; // 当前日期的情感标签值为1的数量

             for (var i = 1; i < lines.length; i++) {
                var parts = lines[i].split(',');
                 var date = parts[0].split(' ')[0]; // 提取日期的格式（例如：7/18）
                 var label = parseInt(parts[1]); // 提取情感标签值，假设在CSV的第五列
                //  console.log(label);

                if (date !== currentDate) {
                    // 新的一天开始，将当天总微博数量添加到数组
                    dates.push(date);
                    totalWeiboCounts.push(totalCount);
                    label0Counts.push(label0Count);
                    label1Counts.push(label1Count);
                    currentDate = date;

                }

                // 累积微博数量
                 totalCount++;
                 if (label === 0) {
                    label0Count++;
                } else if (label === 1) {
                    label1Count++;
                 }
                //  console.log(label0Count);
                //  console.log(label1Count);
            }


             // 最后一天的总微博数量添加到数组
             totalWeiboCounts.push(totalCount);
             label0Counts.push(label0Count);
             label1Counts.push(label1Count);

             // 初始化ECharts配置项
             var option = {
                animationDuration: 45000,
                legend: {
                    data: ['总微博数量', '消极情感的微博数量', '积极情感的微博数量'],
                    textStyle: {
                        color: 'white' // 设置图例文本颜色为白色
                    }
                    // 可以根据需要设置图例的位置和样式等属性
                },
                 tooltip: {
      order: 'valueDesc',
      trigger: 'axis',
        axisPointer: {
        type: 'shadow'
      }
    },
                 xAxis: {
                     type: 'category',
                     data: dates
                 },
                 yAxis: {
                     type: 'value',
                     name: '微博数量'
                 },
                 series: [{
                     name: '总微博数量',
                     type: 'line',
                     data: totalWeiboCounts,
                    //  lineStyle: {
                    //     color: '#66cc66' // 设置折线颜色为红色
                    //   }, areaStyle:{}
                    //  areaStyle: {
                    //      color: 'rgba(0,0,255,0.5)'
                    //  }
                 },
                 {
                    name: '消极情感的微博数量',
                    type: 'line',
                     data: label0Counts,
                    //  areaStyle:{}
                },
                {
                    name: '积极情感的微博数量',
                    type: 'line',
                    data: label1Counts,
                    // areaStyle:{}
                     },
                 ]
             };

             // 使用配置项绘制折线图
             myChart.setOption(option);
         })
    .catch(error => console.error('Error:', error));
         
    