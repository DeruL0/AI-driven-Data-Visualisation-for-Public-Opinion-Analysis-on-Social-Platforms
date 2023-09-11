
    let mapChart;
    let lineChart;
    let rawData;

    async function fetchCSVData(url) {
        const response = await fetch(url);
        const dataBuffer = await response.arrayBuffer();
        const decoder = new TextDecoder("GBK");
        const csvData = decoder.decode(dataBuffer);
        const workbook = XLSX.read(csvData, { type: "string" });
        const sheet = workbook.Sheets[workbook.SheetNames[0]];
        return XLSX.utils.sheet_to_json(sheet);
    }

    function updateMapOption(dataType) {
        // 根据 dataType 选择相应的数据列名
        const dataColumnName = dataType === 'nursingHomes' ? 'nursingHomes' : 'beds';
        const titleText = dataType === 'nursingHomes' ? '各省养老院数量' : '各省养老院床位数量';
        const unit = dataType === 'nursingHomes' ? '家' : '个';

        const data = rawData.map(item => ({
            name: item.province,
            value: item[dataColumnName]
        }));

        if (!mapChart) {
            mapChart = echarts.init(document.getElementById('map'));
        }
        const option = {
            title: {
                text: titleText,
                left: 'center',
                textStyle: {
                    fontSize: 20,
                    color: '#fff',
                }
            },
            tooltip: {
                trigger: 'item',
                formatter: function (params) {
                    return params.name + ': ' + params.value + unit;
                }
            },
            visualMap: {
                min: 0,
                max: Math.max(...data.map(item => item.value)),
                left: 'left',
                top: 'bottom',
                text: ['高', '低'],
                inRange: {
                    color: ['#e0ffff', '#006edd']
                },
                calculable: true,
            },
            series: [
                {
                    name: titleText,
                    type: 'map',
                    map: 'china',
                    roam: false,
                    label: {
                        show: false,
                        fontSize: 8
                    },
                    selectedMode:false,
                    data: data,
                    emphasis: {
                        scale: true,
                        itemStyle: {
                            areaColor: null,
                            borderColor: '#000',
                            borderWidth: 1
                        },
                        label: {
                            show: true,
                            fontSize: 12,
                            position: 'inside',
                        }
                    }
                }
            ]
        };

        mapChart.setOption(option);
        window.addEventListener('resize', function () { mapChart.resize(); });

        // 添加点击事件
        mapChart.on('click', async function (params) {
            const provinceData = await fetchCSVData("datasets/2019至2022各季度各省养老院与床位数据.csv");
            const provinceTrendData = provinceData.filter(item => item.province === params.name);
            await showLineChart(params.name, dataType);
        });

        async function showLineChart(province, dataType) {
            const dataColumnName = dataType === 'nursingHomes' ? 'nursingHomes' : 'beds';
            const titleText = dataType === 'nursingHomes' ? '养老院数量' : '养老院床位数量';
            const unit = dataType === 'nursingHomes' ? '家' : '个';

            const provinceData = await fetchCSVData("datasets/2019至2022各季度各省养老院与床位数据.csv");
            const provinceTrendData = provinceData.filter(item => item.province === province);

            if (provinceTrendData.length > 0) {
                const quarters = provinceTrendData.map(item => item.quarter);
                const values = provinceTrendData.map(item => item[dataColumnName]);

                const minValue = Math.floor(Math.min(...values) / 100) * 100;

                const lineChartContainer = document.getElementById('lineChartContainer');
                lineChartContainer.style.display = 'block';

                // 如果已经存在一个实例，清除这个实例
                if (lineChart) {
                    lineChart.dispose();
                }

                lineChart = echarts.init(document.getElementById('lineChart'));

                const lineOption = {
                    title: {
                        text: province + '各季度' + titleText,
                        left: 'center',
                        textStyle: {
                            color:'#fff'
                        }
                    },
                    tooltip: {
                        trigger: 'axis',
                        formatter: function (params) {
                            return params[0].axisValue + '<br/>' + params[0].seriesName + ': ' + params[0].data + unit;
                        }
                    },
                    xAxis: {
                        type: 'category',
                        name: '时间',
                        data: quarters,
                        axisLabel: {
                            color: '#fff',
                        },
                        nameTextStyle: {
                            color: '#fff',
                        },
                    },
                    yAxis: {
                        type: 'value',
                        name: '数量',
                        min: minValue,
                        axisLabel: {
                            color: '#fff',
                        },
                        nameTextStyle: {
                            color: '#fff',
                        },
                        axisLine:{
                            show:true
                        },
                        axisTick:{show:true},
                        splitLine: {
                            show: true,
                            lineStyle: {
                                color: '#17273B',
                                width: 1,
                                type: 'solid',
                            }
                        }
                    },
                    series: [
                        {
                            name: titleText,
                            type: 'line',
                            data: values
                        }
                    ]
                };
                lineChart.setOption(lineOption);
            }
        }

        const closeButton = document.getElementById('closeButton');
        closeButton.addEventListener('click', function () {
            const lineChartContainer = document.getElementById('lineChartContainer');
            lineChartContainer.style.display = 'none';
        });
    }

    async function createMap() {
        rawData = await fetchCSVData("datasets/最新季度养老院及床位数量.csv");
        updateMapOption('nursingHomes');
    }

    createMap(); // 默认显示养老院数据


    // 拖动折线图窗口
    (function () {
        const lineChartContainer = document.getElementById('lineChartContainer');
        const resizeWrapper = document.getElementById('resizeWrapper');
        let isMouseDown = false;
        let isResizing = false;
        let offsetX = 0;
        let offsetY = 0;

        lineChartContainer.addEventListener('mousedown', function (e) {
            if (e.target === resizeWrapper) {
                isResizing = true;
            } else {
                isMouseDown = true;
            }
            offsetX = e.clientX - lineChartContainer.offsetLeft;
            offsetY = e.clientY - lineChartContainer.offsetTop;
        });

        document.addEventListener('mousemove', function (e) {
            if (isMouseDown) {
                lineChartContainer.style.left = (e.clientX - offsetX) + 'px';
                lineChartContainer.style.top = (e.clientY - offsetY) + 'px';
            } else if (isResizing) {
                lineChartContainer.style.width = (e.clientX - lineChartContainer.offsetLeft) + 'px';
                lineChartContainer.style.height = (e.clientY - lineChartContainer.offsetTop) + 'px';
                lineChart.resize();
            }
        });

        document.addEventListener('mouseup', function () {
            isMouseDown = false;
            isResizing = false;
        });
    })();
    document.getElementById('nursingHomeBtn').addEventListener('click', function () {
        updateMapOption('nursingHomes');
    });

    document.getElementById('bedsBtn').addEventListener('click', function () {
        updateMapOption('beds');
    });


