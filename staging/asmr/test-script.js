/**
 * TEST SCRIPT - Quick demo of all features
 */

module.exports = [
    {
        text: "Normal style",
        pause: 3000,
        style: 'normal'
    },
    {
        type: 'pause',
        duration: 8000,
        label: 'breath here'
    },
    {
        text: "Whisper style\n(quiet, dim)",
        pause: 3000,
        style: 'whisper'
    },
    {
        type: 'pause',
        duration: 6000,
        label: 'clicking'
    },
    {
        text: "SCREAM STYLE\n(BOLD RED)",
        pause: 3000,
        style: 'scream'
    },
    {
        type: 'pause',
        duration: 10000,
        label: 'long pause demo'
    },
    {
        text: "Test complete!\nNotice the auto-scroll.",
        pause: 3000,
        style: 'emphasis'
    }
];
