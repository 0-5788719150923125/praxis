#!/usr/bin/env node
/**
 * ASMR Script Runner - Song Lyrics Style
 *
 * Auto-scrolling display with long dramatic pauses.
 * Lines accumulate on screen like song lyrics.
 */

const readline = require('readline');

class ScriptRunner {
    constructor(script) {
        this.script = script;
        this.displayedLines = [];
        this.rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });
    }

    async run() {
        console.clear();
        console.log('\x1b[2m═══════════════════════════════════════════════════════════\x1b[0m');
        console.log('\x1b[1m              ASMR SCRIPT RUNNER v2.0\x1b[0m');
        console.log('\x1b[2m═══════════════════════════════════════════════════════════\x1b[0m\n');
        console.log('\x1b[33mAuto-scrolling with long pauses for performance\x1b[0m');
        console.log('\x1b[33mPress Ctrl+C to exit\x1b[0m\n');

        await this.sleep(3000);

        for (const line of this.script) {
            await this.displayLine(line);
        }

        console.log('\n\x1b[2m═══════════════════════════════════════════════════════════\x1b[0m');
        console.log('\x1b[32m✓ Script complete\x1b[0m');
        console.log('\x1b[2m═══════════════════════════════════════════════════════════\x1b[0m\n');

        this.rl.close();
        process.exit(0);
    }

    async displayLine(line) {
        if (line.type === 'pause') {
            // Visual pause indicator
            await this.showPauseIndicator(line.duration, line.label);
            return;
        }

        const { text, pause = 5000, style = 'normal' } = line;

        // Apply styling
        let styledText = text;
        switch(style) {
            case 'whisper':
                styledText = `\x1b[2m${text}\x1b[0m`; // Dim
                break;
            case 'scream':
                styledText = `\x1b[1m\x1b[31m${text}\x1b[0m`; // Bold red
                break;
            case 'glitch':
                styledText = `\x1b[5m\x1b[36m${text}\x1b[0m`; // Blinking cyan
                break;
            case 'static':
                styledText = `\x1b[2m\x1b[37m${text}\x1b[0m`; // Dim white
                break;
            case 'emphasis':
                styledText = `\x1b[1m${text}\x1b[0m`; // Bold
                break;
            case 'title':
                styledText = `\x1b[1m\x1b[36m${text}\x1b[0m`; // Bold cyan
                break;
            default:
                styledText = text;
        }

        // Add line to display with proper spacing
        this.displayedLines.push('');
        this.displayedLines.push(styledText);

        // Redraw screen with all accumulated lines
        this.redrawScreen();

        // Pause after displaying
        await this.sleep(pause);
    }

    async showPauseIndicator(duration, label = 'breath') {
        // Show pause marker
        const pauseText = `\x1b[2m\x1b[90m[ ${label} - ${duration/1000}s ]\x1b[0m`;
        this.displayedLines.push('');
        this.displayedLines.push(pauseText);

        this.redrawScreen();

        // Animate the pause with dots
        const steps = Math.floor(duration / 2000); // Update every 2 seconds
        for (let i = 0; i < steps; i++) {
            await this.sleep(2000);
            // Add a dim dot to show time passing
            process.stdout.write('\x1b[2m.\x1b[0m');
        }

        // Final wait for remaining time
        const remaining = duration % 2000;
        if (remaining > 0) {
            await this.sleep(remaining);
        }

        console.log(''); // New line after pause
    }

    redrawScreen() {
        console.clear();

        // Calculate how many lines to show (keep last N lines visible)
        const maxVisibleLines = process.stdout.rows - 5 || 40;
        const startIndex = Math.max(0, this.displayedLines.length - maxVisibleLines);
        const visibleLines = this.displayedLines.slice(startIndex);

        console.log('\n');
        visibleLines.forEach(line => {
            console.log(line);
        });
        console.log('\n');
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

module.exports = ScriptRunner;

// If run directly, load and run the default script
if (require.main === module) {
    const path = require('path');
    const scriptArg = process.argv[2] || './script.js';

    // Resolve script path relative to current working directory
    const scriptPath = path.resolve(process.cwd(), scriptArg);

    try {
        const script = require(scriptPath);
        const runner = new ScriptRunner(script);
        runner.run().catch(err => {
            console.error('\x1b[31mError:\x1b[0m', err.message);
            process.exit(1);
        });
    } catch (err) {
        console.error('\x1b[31mFailed to load script:\x1b[0m', err.message);
        console.error('\x1b[33mUsage:\x1b[0m node runner.js [script.js]');
        process.exit(1);
    }
}
