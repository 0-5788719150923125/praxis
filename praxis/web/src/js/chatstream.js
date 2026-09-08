/**
 * Praxis Web - a streaming assistant turn
 *
 * The model decodes inside the training loop, one request per step, so a turn
 * takes as long as it takes whether or not anyone is watching. Streaming does
 * not change that - it changes a 30-60s blank wait into text appearing, which
 * is the whole difference between "thinking" and "broken".
 *
 * The turn's final content always comes from the POST's own answer, never from
 * the accumulated deltas. They should agree (server-side, both go through the
 * same reply extractor), and settling on the authoritative one anyway is what
 * makes a dropped frame or a socket that was down cost nothing.
 */

import { state } from './state.js';
import { render } from './render.js';

/**
 * Begin an assistant turn that fills in as the reply arrives.
 *
 * Nothing is pushed into `state.messages` until the first delta lands, so the
 * conversation sent to the API never contains the turn being generated, and a
 * request that streams nothing behaves exactly as it did before.
 *
 * @returns {{onDelta: function, onReset: function, settle: function, discard: function}}
 */
export function streamingTurn() {
    let turn = null;
    let pendingFrame = null;

    /** Coalesce renders to one per frame: a byte-level model emits a delta per
     *  byte, and re-serializing the whole message list that often is enough to
     *  drop frames on its own. */
    const scheduleRender = () => {
        if (pendingFrame !== null) return;
        pendingFrame = requestAnimationFrame(() => {
            pendingFrame = null;
            render();
        });
    };

    const ensureTurn = () => {
        if (!turn) {
            turn = { role: 'assistant', content: '', streaming: true };
            state.messages.push(turn);
            // Text is arriving, so the thinking dots have done their job.
            state.isThinking = false;
        }
        return turn;
    };

    return {
        onDelta(text) {
            ensureTurn().content += text;
            scheduleRender();
        },

        /** The runtime spliced a tool result and moved the turn anchor past it:
         *  everything shown so far has stopped being part of the answer. */
        onReset() {
            if (turn) {
                turn.content = '';
                scheduleRender();
            }
        },

        /** Replace the streamed text with the authoritative reply. */
        settle(content) {
            if (turn) {
                turn.content = content;
                delete turn.streaming;
                turn = null;
            } else {
                state.messages.push({ role: 'assistant', content });
            }
        },

        /** Drop the partial turn - the caller is about to report an error. */
        discard() {
            if (!turn) return;
            const index = state.messages.indexOf(turn);
            if (index !== -1) state.messages.splice(index, 1);
            turn = null;
        }
    };
}
