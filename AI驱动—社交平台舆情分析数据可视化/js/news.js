const newsList = document.getElementById('news-list');
let newsData = [];
let currentNewsIndex = 0;
let newsDisplayed = 0;

function fetchNews() {
    fetch('datasets/weibo.csv',{ headers: { 'Content-Type': 'text/csv;charset=utf-8' } })
        .then(response => response.arrayBuffer())
        .then(buffer => {
            const decoder = new TextDecoder('utf-8');
            const csvText = decoder.decode(buffer);
            const parsedData = XLSX.read(csvText, { type: 'string', cellDates: true });
            const wsname = parsedData.SheetNames[0];
            const ws = parsedData.Sheets[wsname];
            newsData = XLSX.utils.sheet_to_json(ws, { header: 1 });
            newsData.shift();
            for (let i = 0; i < 12; i++) {
                displayNews();
            }
        });
}

function displayNews() {
    if (newsDisplayed >= 12) {
        newsList.removeChild(newsList.firstChild);
        newsDisplayed--;
    }

    if (currentNewsIndex >= newsData.length) {
        currentNewsIndex = 0;
    }

    const newsItem = document.createElement('li');
    const newsLink = document.createElement('a');
    newsLink.href = newsData[currentNewsIndex][1];
    newsLink.textContent = newsData[currentNewsIndex][0];
    newsLink.classList.add('news-item');

    if (currentNewsIndex % 2 === 1) {
        newsItem.style.backgroundColor = 'rgba(0, 245, 249, 0.35)';
    }

    newsItem.appendChild(newsLink);
    newsList.appendChild(newsItem);

    currentNewsIndex++;
    newsDisplayed++;
}

setInterval(() => {
    displayNews();
}, 5000);

fetchNews();