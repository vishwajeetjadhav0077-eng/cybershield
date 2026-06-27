let chart = null;

async function loadData() {

    const response = await fetch("/api/stats");
    const data = await response.json();

    document.getElementById("total").textContent = data.total;
    document.getElementById("cyber").textContent = data.cyberbullying;
    document.getElementById("safe").textContent = data.safe;
    document.getElementById("high").textContent = data.high;
    document.getElementById("medium").textContent = data.medium;
    document.getElementById("low").textContent = data.low;

    const tbody = document.getElementById("history");
    tbody.innerHTML = "";

    data.recent.forEach(item => {

        tbody.innerHTML += `
        <tr>
            <td>${item[0]}</td>
            <td>${item[1]}</td>
            <td>${item[2]}%</td>
            <td>${item[3]}</td>
            <td>${item[4]}</td>
        </tr>
        `;

    });

    const ctx = document.getElementById("chart").getContext("2d");

    if (chart) {
        chart.destroy();
    }

    chart = new Chart(ctx, {

        type: "pie",

        data: {

            labels: ["Safe", "Cyberbullying"],

            datasets: [{

                label: "Comments",

                data: [
                    data.safe,
                    data.cyberbullying
                ]

            }]

        },

        options: {

            responsive: true,

            maintainAspectRatio: false

        }

    });

}

loadData();

setInterval(loadData, 5000);