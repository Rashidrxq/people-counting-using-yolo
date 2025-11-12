# near top of file, add:
from flask import render_template_string, url_for


# Add this HTML template string (under config constants)
UPLOAD_PAGE = """
<!doctype html>
<title>People Counter - Upload</title>
<h1>Upload a video for people counting</h1>
<form method="post" action="/upload" enctype="multipart/form-data">
  <input type="file" name="file" accept="video/*" required>
  <input type="submit" value="Upload & Process">
</form>
<p>After processing you'll get a JSON response with counts and a download link.</p>
"""

# Add this route function (below your other routes)
@app.route("/download/<path:filename>", methods=["GET"])
def download_file(filename):
    # secure: ensure file exists in OUTPUT_DIR
    if not os.path.exists(os.path.join(OUTPUT_DIR, filename)):
        abort(404)
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)