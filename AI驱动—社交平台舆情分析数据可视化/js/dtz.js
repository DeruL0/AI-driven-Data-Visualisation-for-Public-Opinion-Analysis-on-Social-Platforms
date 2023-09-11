        // 初始化 ECharts 实例
        const myChart = echarts.init(document.getElementById('dtz'));

        // 使用 fetchCSVData 函数获取 CSV 数据
        async function fetchCSVData(url) {
            const response = await fetch(url);
            const dataBuffer = await response.arrayBuffer();
            const decoder = new TextDecoder("utf-8");
            const csvData = decoder.decode(dataBuffer);
            return csvData;
}
        // 处理 CSV 数据并创建图表
fetchCSVData('/datasets/forward.csv').then(function (csvData) {
    // 将 CSV 数据解析为数组
    var lines = csvData.split('\n');
    var dateList = [];
    var likeData = [];
    var commentData = [];
    var forwardData = [];

    // 遍历 CSV 行数据
    for (var i = 1; i < lines.length; i++) { // 从第二行开始，跳过第一行
        var line = lines[i];
        var parts = line.split(','); // 假设 CSV 使用逗号分隔
        var date = parts[0]; // 第一列是日期
        var like = parseInt(parts[1]); // 第二列是点赞
        var comment = parseInt(parts[2]); // 第三列是评论
        var forward = parseInt(parts[3]); // 第四列是转发  ，制作错误，第二列是转发，第三列是评论，第四列是点赞

        // 处理日期格式，只保留月/日部分
        var dateParts = date.split(' ')[0].split('/');
        var formattedDate = dateParts[1] + '/' + dateParts[2];

        dateList.push(formattedDate);
        likeData.push(like);
        commentData.push(comment);
        forwardData.push(forward);
    }

    // 使用 ECharts 创建折线图
    // var option = {
    //     // title: {
    //     //     text: '点赞、评论、转发数量统计'
    //     // },
    //     legend: {
    //         data: ['转发', '评论', '点赞']
    //     },
    //     xAxis: {
    //         type: 'category',
    //         data: dateList
    //     },
    //     yAxis: {
    //         type: 'value',
    //         name: '数量'
    //     },
    //     series: [
    //         {
    //             name: '转发',
    //             type: 'line',
    //             data: likeData
    //         },
    //         {
    //             name: '评论',
    //             type: 'line',
    //             data: commentData
    //         },
    //         {
    //             name: '点赞',
    //             type: 'line',
    //             data: forwardData
    //         }
    //     ]
    // };

    var option = {
        tooltip: {
          trigger: 'axis',
          axisPointer: {
            type: 'cross',
            label: {
              backgroundColor: '#6a7985'
            }
          }
        },
        legend: {
        //   data: ['Email', 'Union Ads', 'Video Ads', 'Direct', 'Search Engine']
            data: ['转发', '评论', '点赞'],
            textStyle: {  // 设置图例文字样式
                color: 'white'  // 将图例文字颜色设为白色
            }
        },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '3%',
          containLabel: true
        },
        xAxis: [
          {
            type: 'category',
            data: dateList
          }
        ],
        yAxis: [
          {
                type: 'value',
              name: '数量'
          }
        ],
        series: [
          {
            name: '转发',
            type: 'line',
            stack: 'Total',
            areaStyle: {},
            emphasis: {
              focus: 'series'
            },
            data: likeData
          },
          {
            name: '评论',
            type: 'line',
            stack: 'Total',
            areaStyle: {},
            emphasis: {
              focus: 'series'
            },
            data: commentData
          },
          {
            name: '点赞',
            type: 'line',
            stack: 'Total',
            areaStyle: {},
            emphasis: {
              focus: 'series'
            },
            data: forwardData
          }
        ]
      };
    // 使用刚指定的配置项和数据显示图表。
    myChart.setOption(option);
});
