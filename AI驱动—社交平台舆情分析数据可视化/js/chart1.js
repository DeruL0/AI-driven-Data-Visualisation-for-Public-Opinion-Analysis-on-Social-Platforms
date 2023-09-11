async function fetchData() {
    const response = await fetch('datasets/pie3.csv');
    const dataBuffer = await response.arrayBuffer();
    const decoder = new TextDecoder("GBK");
    const csvData = decoder.decode(dataBuffer);
    const workbook = XLSX.read(csvData, { type: "string" });
    const sheet = workbook.Sheets[workbook.SheetNames[0]];
    return XLSX.utils.sheet_to_json(sheet);
}

fetchData().then(data => {

    const chart = echarts.init(document.getElementById('chart1'));
    const male = data.filter(item => item.gender === 'male');
    const option = {
        title: {
            text: '用户微博认证类型',
            subtext: '展现微博用户的认证类型',
            left: 'left',
            top: 0,
            textStyle: {
                fontSize: 15,
                color: '#fff',
            }
        },
        
        tooltip: {
            trigger: 'item',
            formatter: '{a} <br/>{b}: {c}%'
        },
        // legend: {
        //     orient: 'vertical',
        //     left: 'left'
        // },
        series: [
            {
                name: '微博认证类型',
                type: 'pie',
                radius: '50%',
                data: male.map(item => ({ name: item.type, value: item.percentage }))

                
            }
        ]
    };

    chart.setOption(option);
    window.addEventListener('resize', function () { chart.resize(); });
});
