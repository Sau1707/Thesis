
async function expand() {
    let current_scroll = 0;

    // Scroll the page until the bottom is reached
    while (current_scroll <= document.body.scrollHeight) {
        console.log(current_scroll);
        console.log(document.body.scrollHeight);
        current_scroll += 25;
        window.scrollTo(0, current_scroll);
        await new Promise(resolve => setTimeout(resolve, 5));
    }

    console.log('Scrolled to the bottom of the page');
}

var callback = arguments[arguments.length - 1];
expand().then((r) => { callback(r) });