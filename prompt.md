# API /messages Endpoint Debug Task

## Original Problem
The `/messages` endpoint returns `{"error": "'generator'"}` when called via `test.sh`. The script sends a POST request with messages in roles: system, developer, and user.

## Current Status
- Created a minimal `/messages` endpoint that just acknowledges receipt (lines 1045-1073 in api.py)
- Removed all `traceback.print_exc()` calls that were causing secondary BrokenPipeError issues
- The Flask app.config["generator"] is not being set properly when the endpoint is accessed

## Key Issues Discovered
1. Flask app.config doesn't have 'generator' key when /messages is called
2. The error line numbers in tracebacks don't match the actual code (stale bytecode/caching issue)
3. The serve_static function (line 1197) redirects to generate_messages() correctly, but errors still occur

## Failed Approaches
- Moving app.config setup from thread to start() method - didn't fix it
- Clearing Python cache - didn't fix it
- Creating minimal endpoint - code works but server needs restart to pick up changes

## Next Steps - USE A STRUCTURED APPROACH
1. First, verify the server is actually restarted and running the new code
2. Add debug logging to track when/where app.config is set and accessed
3. Check if Flask app instance is being recreated or if there are multiple app instances
4. Consider that the generator might not be initialized when the API server starts
5. Test with a simple health check endpoint first to verify app.config state

**Important**: The user deleted the complex /messages implementation. Keep it simple. Don't add complexity back. Focus on getting the minimal version working first, then gradually add functionality.

The test script is in `test.sh`. The API server code is in `api.py`.