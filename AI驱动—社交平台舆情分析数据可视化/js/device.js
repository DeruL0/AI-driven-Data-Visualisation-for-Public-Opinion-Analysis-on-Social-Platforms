var myhart = echarts.init(document.getElementById('device'));

// 使用 fetchCSVData 函数获取 CSV 数据
async function fetchCSVData(url) {
    const response = await fetch(url);
    const dataBuffer = await response.arrayBuffer();
    const decoder = new TextDecoder("UTF-8");
    const csvData = decoder.decode(dataBuffer);
    return csvData;
}

// 处理 CSV 数据并创建饼图
fetchCSVData('/datasets/device.csv').then(function (csvData) {
    // 将 CSV 数据解析为数组
    var lines = csvData.split('\n');
    var labelCounts = {};

    // 遍历 CSV 行数据
    for (var i = 1; i < lines.length; i++) { // 从第二行开始，跳过第一行
        var line = lines[i];
        var parts = line.split(','); // 假设 CSV 使用逗号分隔
        var label = parts[0].trim(); // 获取标签值并去除空格

        // 忽略标签值出现频率小于等于1的情况
        if (labelCounts[label]) {
            labelCounts[label]++;
        } else {
            labelCounts[label] = 1;
        }
    }

 // 过滤出现频率大于等于10的标签和对应的计数
 var filteredLabels = [];
 var filteredCounts = [];
 var otherCount = 0; // 统计“其他”标签的计数
 for (var label in labelCounts) {
     if (labelCounts[label] >= 500) {
         filteredLabels.push(label);
         filteredCounts.push(labelCounts[label]);
     } else {
         otherCount += labelCounts[label];
     }
 }

 // 如果有“其他”标签，添加到数据中
 if (otherCount > 0) {
     filteredLabels.push('其他');
     filteredCounts.push(otherCount);
 }

    // 使用 ECharts 创建饼图
    var option = {
        tooltip: {
            trigger: 'item',
            formatter: '{a} <br/>{b}: {d}%'
        },
        title: {
            text: '使用设备',
            subtext: '发布微博的网民使用设备的分布',
            textStyle: {  // 设置图例文字样式
                color: 'white'  // 将图例文字颜色设为白色
            }
        },
        series: [
            {
                name: '使用设备类型',
                type: 'pie',
                radius: '55%',
                center: ['50%', '60%'],
                data: filteredLabels.map(function (label, index) {
                    return {
                        name: label,
                        value: filteredCounts[index],
                        textStyle: {
                            color: 'white'
                        }
                    };
                }),
                
            }
        ]
    };

    // 使用刚指定的配置项和数据显示图表。
    myhart.setOption(option);
});