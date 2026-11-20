// Run in the Cosmos parent-directory page's browser console after loading all folders.
(() => {
    "use strict";

    const inputRoot = "/shares/UDC/RnR/MaiProfileV3/AllUsers_V4_Backfill/Denoising";
    const outputRoot = inputRoot;
    const webRoot = "/cosmos/MSN.DnI" + inputRoot;
    const currentUrl = new URL(window.location.href);
    const currentPath = decodeURIComponent(currentUrl.pathname).replace(/\/+$/, "");
    if (currentUrl.hostname !== "www.cosmos09.osdinfra.net" || currentPath !== webRoot) {
        throw new Error("Run this on the cosmos09 MSN.DnI Denoising parent-directory page.");
    }

    const dates = new Set();
    for (const anchor of document.querySelectorAll("a[href]")) {
        const url = new URL(anchor.href, currentUrl);
        if (url.origin !== currentUrl.origin) {
            continue;
        }
        const path = decodeURIComponent(url.pathname);
        if (!path.startsWith(webRoot + "/")) {
            continue;
        }
        const match = /^(\d{4}-\d{2}-\d{2})\/?$/.exec(path.slice(webRoot.length + 1));
        if (!match) {
            continue;
        }
        const date = new Date(match[1] + "T00:00:00Z");
        if (Number.isNaN(date.getTime()) || date.toISOString().slice(0, 10) !== match[1]) {
            console.warn("Ignoring invalid date folder:", match[1]);
            continue;
        }
        dates.add(match[1]);
    }

    if (dates.size === 0) {
        throw new Error("No yyyy-MM-dd folder links found. Load the directory listing first.");
    }

    const sortedDates = [...dates].sort();
    const lines = [
        "// Generated from the folder links loaded on the Cosmos directory page.",
        "// Submit to MSN.DnI on cosmos09. Each input stream has its own output.",
        '#DECLARE InputRoot string = "' + inputRoot + '";',
        '#DECLARE OutputRoot string = "' + outputRoot + '";',
        "",
    ];
    sortedDates.forEach((date, index) => {
        lines.push(
            "// " + date,
            '#DECLARE InputStream' + index + ' string = @InputRoot + "/' + date + '/ForGptLabeling.ss";',
            '#DECLARE OutputStream' + index + ' string = @OutputRoot + "/' + date + '/ForGptLabeling.tsv";',
            "",
            "raw" + index + " = SSTREAM @InputStream" + index + ";",
            "",
            "kept" + index + " =",
            "    SELECT UserId,",
            "           Date,",
            "           Source,",
            "           Action",
            "    FROM raw" + index,
            "    WHERE denoising_label == 1;",
            "",
            "OUTPUT kept" + index,
            "TO @OutputStream" + index,
            "USING DefaultTextOutputter(delimiter: '\\t', outputHeader: true);",
            "",
        );
    });

    const script = lines.join("\n");
    const blob = new Blob([script], { type: "text/plain;charset=utf-8" });
    const downloadUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = "batch_denoised_ss_to_tsv.script";
    document.body.appendChild(link);
    try {
        link.click();
    } finally {
        link.remove();
        setTimeout(() => URL.revokeObjectURL(downloadUrl), 1000);
    }
    console.log("Generated SCOPE outputs for " + sortedDates.length + " streams:", sortedDates);
    console.warn("Only folders loaded on this page are included. Check all folders/pages are loaded before submitting.");
})();
