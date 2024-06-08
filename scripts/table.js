
function extractStockData() {
    // Define an array to hold the data
    let stocksData = [];

    // Locate the 'tbody' within the table with id 'stocks_table' and iterate over each 'tr'
    $('#stocks_table tbody tr').each(function () {
        // For each 'tr', find the 'a' with the specific class and collect href and inner text
        let linkElement = $(this).find('a.link.link--blue.table-child--middle.align-top');
        let href = linkElement.attr('href');
        let text = linkElement.text().replace(/\s+/g, ' ').trim();

        // Find the element with the 'aria-label' "Weight" and get its text
        let weight = $(this).find('[aria-label="Weight"]').text();

        // Append the gathered data to the array as an object
        stocksData.push({
            href: href,
            text: text,
            weight: weight
        });
    });

    // Return the collected data
    return stocksData;
}


return JSON.stringify(extractStockData());