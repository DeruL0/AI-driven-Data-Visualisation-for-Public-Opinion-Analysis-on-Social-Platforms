var mhart = echarts.init(document.getElementById('name'));
async function fetchCSVData(url) {
    const response = await fetch(url);
    const dataBuffer = await response.arrayBuffer();
    const decoder = new TextDecoder("UTF-8");
    const csvData = decoder.decode(dataBuffer);
    return csvData;
}

// 处理 CSV 数据并创建图表
fetchCSVData('/datasets/name.csv').then(function (csvData) {
    // 将 CSV 数据解析为数组
    var lines = csvData.split('\n');
    var labelCounts = {};

    // 遍历 CSV 行数据
    for (var i = 1; i < lines.length; i++) { // 从第二行开始，跳过第一行
        
        var line = lines[i];
        if (line.includes('#NAME?')) {
            continue; // 跳过包含无用标签值的行
        }
        var parts = line.split(','); // 假设 CSV 使用逗号分隔
        var label = parts[0].trim(); // 获取标签并去除空格

        // 统计标签出现的次数
        if (labelCounts[label]) {
            labelCounts[label]++;
        } else {
            labelCounts[label] = 1;
        }
    }

    // 将标签和出现次数转换为数组形式
    var labelArray = Object.keys(labelCounts);
    var countArray = Object.values(labelCounts);

    // 排序标签和次数数组（按次数降序排序）
    var sortedData = labelArray.map(function (label, index) {
        return { label: label, count: countArray[index] };
    }).sort(function (a, b) {
        return b.count - a.count;
    });

    // 选择显示前几个标签（例如前 10 个）
    var topLabels = sortedData.slice(0, 10);

    // 使用 ECharts 创建漏斗图
    var option = {
        title: {
            text: '活跃度最高的账号',
            subtext: '统计了事件发生期间活跃度最高的一些微博账号',
            textStyle: {  // 设置图例文字样式
                color: 'white'  // 将图例文字颜色设为白色
            }
        },
        tooltip: {
            trigger: 'item',
            formatter: '{a} <br/>{b}: {c} ({d}%)'
        },
        series: [
            {
                name: '标签出现次数',
                type: 'funnel',
                left: '10%',
                top: '15%',
                width: '80%',
                min: 0,
                max: topLabels[0].count, // 最大值取最高频次
                minSize: '0%',
                maxSize: '100%',
                sort: 'descending',
                gap: 2,
                label: {
                    show: true,
                    position: 'inside'
                },
                labelLine: {
                    length: 10,
                    lineStyle: {
                        width: 1,
                        type: 'solid'
                    }
                },
                data: topLabels.map(function (item) {
                    return { name: item.label, value: item.count };
                })
            }
        ]
    };

    // 初始化 ECharts 实例并显示图表
    mhart.setOption(option);
});