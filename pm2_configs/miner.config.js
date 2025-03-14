module.exports = {
    apps: [{
        name: "folding-miner-c5-h1",
        script: "scripts/run_miner.sh", // Use the wrapper script
        autorestart: true,
        watch: false
    }]
};
