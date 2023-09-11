var yChart = echarts.init(document.getElementById('content'));

// 使用 fetchCSVData 函数获取 CSV 数据
async function fetchCSVData(url) {
    const response = await fetch(url);
    const dataBuffer = await response.arrayBuffer();
    const decoder = new TextDecoder("UTF-8");
    const csvData = decoder.decode(dataBuffer);
    return csvData;
}

// 处理 CSV 数据并创建堆叠面积图
fetchCSVData('/datasets/forward.csv').then(function (csvData) {
    // 将 CSV 数据解析为数组
    var lines = csvData.split('\n');
    var dateCounts = {}; // 用于存储每一天的频数

    // 遍历 CSV 行数据
    for (var i = 1; i < lines.length; i++) { // 从第二行开始，跳过第一行
        var line = lines[i];
        var parts = line.split(','); // 假设 CSV 使用逗号分隔
        var dateAndTime = parts[0].trim(); // 获取日期时间并去除空格

        // 从日期时间中提取日期部分（月份和日期）
        var dateParts = dateAndTime.split(' ')[0].split('/');
        var formattedDate = dateParts[1] + '/' + dateParts[2];

        // 统计每一天出现的频数
        if (dateCounts[formattedDate]) {
            dateCounts[formattedDate]++;
        } else {
            dateCounts[formattedDate] = 1;
        }
    }

    // 将数据转换为 ECharts 堆叠面积图需要的格式
    var dateList = Object.keys(dateCounts);
    var countList = dateList.map(function (date) {
        return dateCounts[date];
    });

    // 使用 ECharts 创建堆叠面积图
    var option = {
        title: {

            subtext: '统计事件期间每日网民发送微博数量',
            textStyle: {  // 设置图例文字样式
                color: 'white'  // 将图例文字颜色设为白色
            }
        },
                tooltip: {
            trigger: 'item',
            formatter: '{a} <br/>{b}: {c}'
        },
        xAxis: {
            type: 'category',
            data: dateList,
            axisPointer: {
                type: 'cross',
                label: {
                    show: true,
                    formatter: '7/18' // 手动设置原点的位置
                },
                snap: true
            },
            axisLabel: {
                interval: Math.ceil(dateList.length / 10), // 设置标签间隔
                rotate: 45 // 可选，标签旋转角度
            }
        },
        yAxis: {
            type: 'value',
            name: '数量(x1000)',
            axisLabel: {
                formatter: function (value) {
                    // 将每个标签值除以1000，以实现缩小1000倍
                    return (value / 1000); // 可根据需要进行精确度控制
                }
            }
        },
        series: [
            {
                name: '数量',
                type: 'line',
                stack: '总量',
                data: countList
            }
        ]
    };

    // 使用刚指定的配置项和数据显示图表。
    yChart.setOption(option);
});