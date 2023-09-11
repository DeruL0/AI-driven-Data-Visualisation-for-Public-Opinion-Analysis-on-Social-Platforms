// 使用 fetchCSVData 函数获取 CSV 数据
async function fetchCSVData(url) {
    const response = await fetch(url);
    const dataBuffer = await response.arrayBuffer();
    const decoder = new TextDecoder("UTF-8");
    const csvData = decoder.decode(dataBuffer);
    return csvData;
}

// 处理 CSV 数据
fetchCSVData('/datasets/simple.csv').then(function (csvData) {
    // 将 CSV 数据解析为数组
    var lines = csvData.split('\n');
    var timeData = [];
    var heatData = [];
    
    // 遍历 CSV 行数据
    for (var i = 1; i < lines.length; i++) { // 从第二行开始，跳过第一行
        var line = lines[i];
        var parts = line.split(','); // 假设 CSV 使用逗号分隔
        // var date = parts[0].trim().split(' ')[0]; // 获取日期并去除空格
        var date = parts[0].trim();
        var forward = parseInt(parts[1]);
        var comment = parseInt(parts[2]);
        var like = parseInt(parts[3]);
        var certification = parts[4]; // 获取认证类型并去除空格

        // 计算热度值
        var heat = forward * 0.5 + comment * 0.3 + like * 0.2;
        if (certification === '蓝V认证') {
            heat *= 1.3;
        } else if (certification === '红V认证') {
            heat *= 1.2;
        } else if (certification === '黄V认证') {
            heat *= 1.1;
        }

        // 累积热度值
        if (i === 1) {
            heatData.push(heat);
        } else {
            heatData.push(heatData[i - 2] + heat);
        }

        // 保存时间点
        timeData.push(date);
    }

    heatData[0] = 1;
    heatData[1] = 2;
    timeData.pop(); // 删除 timeData 数组的最后一个元素
    heatData.pop(); // 删除 heatData 数组的最后一个元素

    console.log(timeData);
    console.log(heatData);
    // 假设 timeData 是包含日期字符串的数组
    var timeStamps = [];

    for (var i = 0; i < timeData.length; i++) {
        // 将日期字符串转换为时间戳
        var timeStamp = new Date(timeData[i]).getTime();

        // 检查是否成功转换，如果不成功则跳过这一行数据
        if (!isNaN(timeStamp)) {
            timeStamps.push(timeStamp);
        } else {
            console.error("Failed to convert date string:", timeData[i]);
        }
    }
    // console.log(timeStamps);
    // 现在 timeStamps 数组包含了时间戳，可以与热度数据进行矩阵运算
    const timeStampsNumeric = timeStamps.map(timestamp => parseFloat(timestamp) / 100000);
    console.log(timeStampsNumeric);
    const degree = 4;
    // 定义多项式回归函数
    function polynomialRegression(x, y, degree) {
        const X = [];
        for (let i = 0; i < x.length; i++) {
            const row = [];
            for (let j = 0; j <= degree; j++) {
                row.push(Math.pow(x[i], j));
            }
            X.push(row);
        }
        console.log(X);
        // 使用数学库进行矩阵操作
        const XMatrix = math.matrix(X);
        const yMatrix = math.matrix(y);

        // 计算多项式回归系数
        const Xt = math.transpose(XMatrix);
        const XtX = math.multiply(Xt, XMatrix);
        const XtXInverse = math.inv(XtX);
        const XtXInverseXt = math.multiply(XtXInverse, Xt);
        const coefficients = math.multiply(XtXInverseXt, yMatrix);
        console.log(XMatrix);
        console.log(yMatrix);
        // 返回多项式系数
        return coefficients._data;
    }

    // 使用多项式回归函数拟合数据
    const coefficients = polynomialRegression(timeStampsNumeric, heatData, degree);
    console.log(coefficients);

    // 计算拟合曲线的导数值
    const fittedDerivativeData = timeStampsNumeric.map((x) => {
        let result = 0;
        for (let i = 1; i <= degree; i++) {
            result += coefficients[i] * i * Math.pow(x, i - 1);
        }
        return result;
    });
    
    console.log(fittedDerivativeData);

    // 在这里得到了拟合曲线的导数 fittedDerivativeData
    // 现在您可以使用 fittedDerivativeData 在图表中显示热度值曲线

    // 使用 ECharts 创建图表
    var option = {
        animationDuration: 10000,
        tooltip: {
            trigger: 'axis', // 将触发机制改为axis型
        formatter: function (params) {
            // 自定义tooltip的内容
            var result = params[0].name + '<br>';
            params.forEach(function (item) {
                result += item.seriesName + ': ' + item.data + '<br>';
            });
            return result;
        }
        },
        legend: {
            data: ['热度值曲线'], // 修改图例名称
            textStyle: {
                color: 'white' // 设置图例文字颜色为白色
            }
        },
        xAxis: {
            type: 'category',
            data: timeData
        },
        yAxis: {
            name: '热度值',
            type: 'value'
        },
        series: [
            {
                animation: {
                    duration: 2000, // 动画时长，单位毫秒
                    easing: 'linear'
                }, // 缓动效果
                name: '热度值曲线',
                type: 'line',
                data: fittedDerivativeData // 使用热度值曲线数据
            }
        ]
    };

    // 初始化 ECharts 实例并显示图表
    var myChart = echarts.init(document.getElementById('prediction2'));
    myChart.setOption(option);
    // 在这里进行多项式线性拟合的导数计算和绘制图表
})    .catch(error => console.error('Error:', error));;