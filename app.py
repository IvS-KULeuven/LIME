from flask import Flask, request, jsonify, render_template, send_file, url_for
import subprocess
import numpy as np
import os
import threading
import shutil
from mcak_explore import main as mcak_main
from mcak_explore import DUMMY_RESULTS
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from PIL import Image
import pandas as pd
import tempfile
from jinja2 import Template
import csv
from celery import Celery, Task, shared_task
import io
import logging
import signal
from time import time
# from logging.handlers import SysLogHandler
from config import ServerConfig

logging_level = logging.INFO

# Set up the logging of information, warning, and errors
logger = logging.getLogger("LIME_app")
logger.setLevel(logging_level)
formatter = logging.Formatter(
        fmt="%(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - '%(message)s'",
        datefmt="%Y-%m-%d %H:%M:%S"
        )
# For now keep the log file in the main directory
file_handler = logging.FileHandler("LIME_app.log")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging_level)
logger.addHandler(file_handler)

logger.info(f"Starting up LIME! ({__name__})")

# Dummy(ish) syslog for now
# syslog_handler = logging.FileHandler("dummy_syslog")
# syslog_handler.setFormatter(formatter)
# syslog_handler.setLevel(logging_level)


logging.info(f"Using address:{ServerConfig.FLASK_SERVER_NAME}")


def celery_init_app(app: Flask) -> Celery:
    """Initialization of the Celery app
    Taken from https://flask.palletsprojects.com/en/stable/patterns/celery/"""
    class FlaskTask(Task):
        def __call__(self, *args: object, **kwargs: object) -> object:
            with app.app_context():
                return self.run(*args, **kwargs)

    celery_app = Celery(app.name, task_cls=FlaskTask)
    celery_app.config_from_object(app.config["CELERY"])
    celery_app.set_default()
    celery_app.conf.worker_concurrency = ServerConfig.FLASK_WORKERS
    app.extensions["celery"] = celery_app
    return celery_app


# Initialize the website app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Use redis as broker for the queueing system.
app.config.from_mapping(
    CELERY=dict(
        broker_url=f"redis://{ServerConfig.REDIS_HOST}:{ServerConfig.REDIS_PORT}/{ServerConfig.REDIS_INDEX}",
        result_backend=f"redis://{ServerConfig.REDIS_HOST}:{ServerConfig.REDIS_PORT}/{ServerConfig.REDIS_INDEX}",
        task_ignore_result=True,
    ),
)

# Server property setup
app.config["SERVER_NAME"] = ServerConfig.FLASK_SERVER_NAME
app.config["PREFERRED_URL_SCHEME"] = ServerConfig.FLASK_URL_SCHEME
app.config["APPLICATION_ROOT"] = ServerConfig.FLASK_ROOT

# Make Flask log the in the log file
app.logger.addHandler(file_handler)
# app.logger.addHandler(syslog_handler)
app.logger.setLevel(logging_level)

# Start Celery for queueing
celery = celery_init_app(app)

MFORCE_DIR = ServerConfig.MFORCE_DIR
DATA_DIR = os.path.join(MFORCE_DIR, ServerConfig.MFORCE_DATA_SUBDIR)
os.makedirs(DATA_DIR, exist_ok=True)  

# Atomic masses for elements
ATOMIC_MASSES = dict(np.genfromtxt("atom_masses.txt", delimiter=",", dtype=None, encoding="utf"))

base_table_data = {"mass_loss_rate": "-",
                   "terminal_velocity": "-",
                   "gamma_e": "-",
                   "qbar": "-",
                   "alpha": "-",
                   "q0": "-",
                   "info": ""}

UPLOAD_FOLDER = os.path.join(ServerConfig.BASE_TMP_DIR, ServerConfig.UPLOAD_SUBDIR)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {"csv", "pdf"}


@app.route("/send_contact_email", methods=["POST"])
def send_contact_email():
    name = request.form.get("name")
    email_addr = request.form.get("email")
    message = request.form.get("message")
    attachment = request.files.get("attachment")

    if not name or not email_addr or not message:
        logger.warning("Failed contact form entry.")
        return jsonify({"error": "Missing required fields"}), 400

    # Validate attachment (csv and pdf only)
    attachment_path = None
    if attachment and allowed_file(attachment.filename):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        attachment_path = os.path.join(UPLOAD_FOLDER, attachment.filename)
        attachment.save(attachment_path)
    elif attachment and not allowed_file(attachment.filename):
        return jsonify({"success": True, "attachment_oke": False})

    # Render email bodies using Jinja2
    admin_body = render_template("contact_help.j2", name=name, email=email_addr, message=message, attachment=attachment.filename if attachment else None, is_user=False)
    user_body = render_template("contact_help.j2", name=name, email=email_addr, message=message, attachment=attachment.filename if attachment else None, is_user=True)

    # Email subject
    subject_admin = f"New Contact Form Submission from {name}"
    subject_user = "LIME Contact Form - Confirmation"

    # Send email to LIME team
    admin_command = [
        "python3", "./mailing/mailer.py",
        "--t", ServerConfig.ADMIN_MAIL_ADDRESS,
        "--s", subject_admin,
        "--b", admin_body
    ]
    if attachment_path:
        admin_command.extend(["--a", attachment_path])

    subprocess.run(admin_command)

    # Send confirmation email to user
    user_command = [
        "python3", "./mailing/mailer.py",
        "--t", email_addr,
        "--s", subject_user,
        "--b", user_body
    ]
    if attachment_path:
        user_command.extend(["--a", attachment_path])

    subprocess.run(user_command)

    # Cleanup attachment
    if attachment_path:
        os.remove(attachment_path)

    logger.info("Contact form filled in and emails sent!")
    return jsonify({"success": True, "attachment_oke": True})


def sigterm_handler(signum, frame):
    """Small function to deal with fortran crashes (STOP statements)"""
    logger.error(f"Mforce crashed! Got this info: {signum}, {frame}")
    raise RuntimeError(f"Mforce crashed! Got this info: {signum}, {frame}")


# Start the sigterm handler
signal.signal(signal.SIGTERM, sigterm_handler)


def calculate_metallicity_massb(mass_abundances):
    """Calculates the actual metallicity from the number abundances input by the user"""
    metals = {e for e in ATOMIC_MASSES if e not in {'H', 'HE'}}
    metallicity = sum(mass_abundances[element] for element in metals if element in mass_abundances)
    return metallicity


def He_number_abundance(mass_abundances):
    """
    Compute the number abundance of helium from mass abundances.
    This is relative to the Hydrogen abundances.
    
    :return: Number abundance of helium (relative to all elements)
    """
    mass_H = mass_abundances.get('H', 0.0)
    mass_He = mass_abundances.get('HE', 0.0)
    
    total_num_abun = sum(mass_abundances[element] / ATOMIC_MASSES[element] for element in mass_abundances)

    N_H = (mass_H / ATOMIC_MASSES['H']) / total_num_abun
    N_He = (mass_He / ATOMIC_MASSES['HE']) / total_num_abun

    if N_H == 0.0 and N_He == 0.0:
        NHe = 1e-4    
    else:
        NHe = N_He/N_H    
    return NHe


def load_email_body(filename):
    with open(filename, 'r') as file:
        return file.read()


def load_dyn_email(filename, context):
    """Load and render email body from Jinja2 template."""
    with open(filename, 'r') as file:
        template = Template(file.read())
    return template.render(context)


def batch_tracker(num_stars):
    """Adds a line to the batch counter file, with the number of stars requested"""
    with open("batch_tracker.log", "a") as counter:
        counter.write(f"{num_stars}\n")


def get_request_count():
    """Gets the total number of requests so far, return 0 if the file does not exist"""
    try:
        with open("request_counter.log", "r") as f:
            return int(f.read().strip())
    except IOError:
        return 0


def increment_request_count():
    """Increases the total number of requests"""
    count = get_request_count() + 1
    with open("request_counter.log", "w") as f:
        f.write(str(count))


def make_data_dict(results_dict):
    """
    Makes a dictionary to be passed to the HTML side of the code to update the table
    """
    data = {"mass_loss_rate": f"{results_dict['mdot']:.3g} Msun / yr",
            "terminal_velocity": f"{results_dict['vinf']:.0f} km / s",
            "gamma_e": f"{results_dict['Gamma_e']:.2f}",
            "qbar": f"{results_dict['Qbar']:.4g}",
            "alpha": f"{results_dict['alpha']:.2f}",
            "q0": f"{results_dict['Q0']:.4g}",
            "info": f"{results_dict['fail_reason']}"}
    return data


def make_fail_pdf(output_dir, pdf_name, failure_reason):
    """Makes a simple PDF file that shows the reason of the failed calculation"""
    pdf_filename = os.path.join(output_dir, f"{pdf_name}.pdf")
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(220, 700, "Simulation Failed")
    c.setFont("Helvetica", 14)
    c.drawString(100, 650, "Reason for failure:")
    wrapped_text = simpleSplit(failure_reason, "Helvetica", 12, 400)
    y_pos = 620
    for line in wrapped_text:
        c.drawString(120, y_pos, line)
        y_pos -= 20
    c.save()


def make_output_pdf(output_dir, pdf_name, results_dict, abundances, input_parameters):
    """
    Makes a pdf with more detailed output of the results.
    """
    lum, mstar, teff, zstar, zscale = input_parameters

    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lavender),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ])

    pdf_filename = os.path.join(output_dir, f"{pdf_name}.pdf")
    figures_list = [os.path.join(output_dir, f) for f in os.listdir(output_dir)
                    if (f.endswith(".png") and not f.startswith("._"))]

    # Initialize
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    page_width, page_height = letter

    logo_path = "./static/logo_2.png"
    title = "LIME Results"

    if os.path.exists(logo_path):
        c.drawImage(logo_path, 50, page_height - 150, width=120, height=120, preserveAspectRatio=True, anchor='c')

    # Add a warning at the top of the pdf file if there is a potential issue
    warning_path = "./static/warning.png"
    if results_dict["warning"]:
        c.setFont("Helvetica", 10)
        c.drawString(130, page_height - 15, results_dict["warning_message"])
        if os.path.exists(warning_path):
            c.drawImage(warning_path, 113, page_height - 17.5, width=15, height=15, preserveAspectRatio=True,
                        anchor='c')

    c.setFont("Helvetica-Bold", 30)
    c.drawString(220, page_height - 150, title)

    table_data = [
        ("Input Parameter", "Value"),
        ("Luminosity [solar luminosity]", f"{lum:.1f}"),
        ("Stellar Mass [solar mass]", f"{mstar:.1f}"),
        ("Eddington Ratio ", "{:.2f}".format(results_dict["Gamma_e"])),
        ("Stellar Radius [solar radius] ", "{:.2f}".format(results_dict["R_star"])),
        ("log g ", "{:.2f}".format(results_dict["log_g"])),
        ("Effective Temperature [K]", f"{teff:.1f}"),
        ("Z star (Calculated)", f"{zstar:.3e}")]

    c.setFont("Helvetica", 16)
    table = Table(table_data)
    table.setStyle(table_style)
    table.wrapOn(c, page_width, page_height)
    table.drawOn(c, 70, page_height - 350)

    # Make the main results table
    table_data = [("Output", "Value"),
                  ("Mass loss rate [solar mass/year]", "{:.3e}".format(results_dict["mdot"])),
                  ("vinf [km/s]", "{:.2f}".format(results_dict["vinf"])),
                  ("Electron scattering opacity ", "{:.2f}".format(results_dict["kappa_e"])),
                  ("Critical depth", "{:.2f}".format(results_dict["t_crit"])),
                  ("Q bar", "{:.2f}".format(results_dict["Qbar"])),
                  ("alpha", "{:.2f}".format(results_dict["alpha"])),
                  ("Q0", "{:.2f}".format(results_dict["Q0"]))]

    c.setFont("Helvetica", 16)
    table = Table(table_data)
    table.setStyle(table_style)
    table.wrapOn(c, page_width, page_height)
    table.drawOn(c, 70, page_height - 530)

    # Make a table with some more advanced and specific output information
    table_data = [("Extra Output", "Value"),
                  ("Z scaled to solar (input)", f"{zscale:.2e}"),
                  ("Globally fitted alpha", "{:.3f}".format(results_dict["alphag"])),
                  ("Locally fitted alpha", "{:.3f}".format(results_dict["alphal"])),
                  ("Effective v escape [km/s]", "{:.2f}".format(results_dict["vesc"])),
                  ("Critical velocity [km/s]", "{:.2f}".format(results_dict["v_crit"])),
                  ("Critical density [g/cm^3]", "{:.2e}".format(results_dict["density"]))]

    c.setFont("Helvetica", 16)
    table = Table(table_data)
    table.setStyle(table_style)
    table.wrapOn(c, page_width, page_height)
    table.drawOn(c, 70, page_height - 690)

    # Abundances Table
    abundance_table_data = [("Element", "Abundance")] + [(el, f"{abundances[el]:.4e}") for el in abundances]
    abundance_table = Table(abundance_table_data, colWidths=[100, 150])
    abundance_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))

    abundance_table.wrapOn(c, page_width, page_height)
    abundance_table.drawOn(c, 300, page_height - 760)

    # --- 1. Display "sim_log.png" with a title ---
    sim_log_image_path = os.path.join(output_dir, "sim_log.png")
    if sim_log_image_path in figures_list:
        c.showPage()
        c.setFont("Helvetica-Bold", 30)
        c.drawString(150, page_height - 50, "Simulation Log Figures")

        img = Image.open(sim_log_image_path)
        img_width, img_height = img.size
        aspect_ratio = img_width / img_height
        new_width = page_width - 100
        new_height = new_width / aspect_ratio

        if new_height > page_height - 100:
            new_height = page_height - 50
            new_width = new_height * aspect_ratio

        x_position = (page_width - new_width) / 2
        y_position = (page_height - new_height) / 2

        c.drawImage(sim_log_image_path, x_position, y_position, width=new_width, height=new_height)

        c.setFont("Helvetica", 14)
        description = "This figure shows the evolution of various physical parameters over the iterations until convergence."
        text_x, text_y = 50, 80

        wrapped_text = simpleSplit(description, "Helvetica", 14, page_width - 100)
        for line in wrapped_text:
            c.drawString(text_x, text_y, line)
            text_y -= 16

        figures_list.remove(sim_log_image_path)

    # --- 2. Arrange Remaining Figures in Pages with Titles ---
    images_per_row, images_per_col = 2, 3
    images_per_page = images_per_row * images_per_col
    max_width = (page_width - 100) / images_per_row
    max_height = (page_height - 150) / images_per_col
    x_start, y_start = 50, page_height - 300

    for i, fig_path in enumerate(figures_list):
        if i % images_per_page == 0:
            c.showPage()

        c.setFont("Helvetica-Bold", 30)
        c.drawString(140, page_height - 50, "Line force multiplier v t")

        row = (i % images_per_page) // images_per_row
        col = (i % images_per_page) % images_per_row
        x_position = x_start + col * max_width
        y_position = y_start - row * max_height

        # Load image to get actual dimensions
        with Image.open(fig_path) as img:
            img_width, img_height = img.size

        # Compute scaling factor while maintaining aspect ratio
        scale_factor = min(max_width / img_width, max_height / img_height)
        new_width = img_width * scale_factor
        new_height = img_height * scale_factor

        # Center the image within its allocated space
        x_adjusted = x_position + (max_width - new_width) / 2
        y_adjusted = y_position + (max_height - new_height) / 2

        c.drawImage(fig_path, x_adjusted, y_adjusted, width=new_width, height=new_height)

        c.setFont("Helvetica", 10)

    c.save()


def process_computation(lum, teff, mstar, zscale, zstar, helium_abundance, abundances,
                        batch_output_dir, does_plot):
    """Runs mcak_explore and generates a pdf with the results if desired. """

    try:
        output_dir = os.path.join(batch_output_dir, "result")
        os.makedirs(output_dir, exist_ok=True)
        massabun_loc = os.path.join(output_dir, "output")
        os.makedirs(massabun_loc, exist_ok=True)
        abundance_filename = os.path.join(massabun_loc, "mass_abundance")
        
        with open(abundance_filename, "w") as f:
            for i, (element, value) in enumerate(abundances.items(), start=1):
                f.write(f"{i:2d}  '{element.upper():2s}'   {value:.14f}\n")

        # Run the main calculation!
        start = time()
        logger.info(f"Starting calculation with L={lum:.3g}, T={teff:.3g}, M={mstar:.3g}, Z={zstar:.3g}")
        results_dict = mcak_main(lum, teff, mstar, zstar, helium_abundance, output_dir, does_plot, logger=logger)
        logger.info(f"Calculation done! It took {results_dict['Iteration']} iterations in {time() - start:.2f} seconds")
        if results_dict["fail"]:
            failure_reason = results_dict["fail_reason"]
            logger.info(f"Computation failed: {failure_reason}")

            if does_plot:
                make_fail_pdf(output_dir, "result", failure_reason)
                logger.info("Made a failed calculation pdf")
            return results_dict

        # make some diagnostic plots and informative tables for the pdf output.
        if does_plot:
            logger.info("Making result pdf")
            input_parameters = [lum, mstar, teff, zstar, zscale]
            make_output_pdf(output_dir, "result", results_dict, abundances, input_parameters)
            logger.info("PDF made!")
        return results_dict

    except Exception as e:
        error_message = str(e)
        logger.error(f"Unexpected error: {error_message}")

    results_dict = dict(DUMMY_RESULTS)
    results_dict["fail_reason"] = f"Unknown crash! Make sure input is physical. "

    return results_dict


@app.route('/')
def home():
    return render_template("index.html", data=base_table_data)


@app.route('/process_data', methods=['POST'])
def start_process():
    """This starts the data processing task"""
    logger.info("Receiving single calculation request")
    data = request.json

    # Check for issues before putting calculations in the queue
    teff = float(data.get("teff", 0.0))
    if teff > 60000 or teff < 18000:
        logger.error("Incorrect temperature selected!")
        return jsonify({"error": "Temperature beyond current coverage"}), 400

    abundances = data.get("abundances", {})
    if not abundances:
        logger.error("Abundances are missing!")
        return jsonify({"error": "Abundances data is missing"}), 400

    task = process_data.apply_async(args=[data])
    logger.info(f"Queued single calculation task with task_id: {task.id}")
    return jsonify({"task_id": task.id}), 202


@shared_task(ignore_result=False)
def process_data(data):
    """Handles the communication with index.html"""
    logger.info("Starting model calculation")
    # Keep track of the number of requests
    increment_request_count()
    try:
        # Extract parameters
        luminosity = float(data.get("luminosity", 0.0))
        teff = float(data.get("teff", 0.0))

        mstar = float(data.get("mstar", 0.0))
        zscale = float(data.get("zscale", 0.0))
        abundances = data.get("abundances", {})
        
        recipient_email = data.get("email", "").strip()

        # To allow additional output:
        expert_mode = data.get("expert_mode", False)  # by default false

        # Always send the expert mode output per email.
        if recipient_email != "":
            expert_mode = True
        
        # Calculate the metallicity and helium number abundance based on the input mass fractions
        zstar = calculate_metallicity_massb(abundances)
        helium_abundance = He_number_abundance(abundances)

        # Create a temporary directory
        base_tmp_dir = ServerConfig.BASE_TMP_DIR
        os.makedirs(base_tmp_dir, exist_ok=True)
        session_tmp_dir = tempfile.mkdtemp(dir=base_tmp_dir)

        pdf_name = "result"
        output_dir = os.path.join(session_tmp_dir, pdf_name)  
        os.makedirs(output_dir, exist_ok=True)  
        pdf_path = os.path.join(output_dir, f"{pdf_name}.pdf")  

        # In individual calculations always make the verification plots
        if expert_mode:
            does_plot = True
        else:
            does_plot = False

        # Run computation in a separate thread
        results_dict = process_computation(luminosity, teff, mstar, zscale, zstar, helium_abundance, abundances,
                                           session_tmp_dir, does_plot)

        table_data = make_data_dict(results_dict)

        # Check if the PDF exists at the correct path
        if expert_mode and not os.path.exists(pdf_path):
            logger.error(f"Did not find the PDF at {pdf_path}")
            with app.app_context():
                return {"error": f"PDF generation failed. Expected at {pdf_path}", **base_table_data}, 500

        # Schedule cleanup
        threading.Timer(20, shutil.rmtree, args=[session_tmp_dir], kwargs={"ignore_errors": True}).start()

        if expert_mode:
            # Generate a downloadable link
            relative_session_id = os.path.relpath(session_tmp_dir, base_tmp_dir)
            pdf_url = url_for("download_temp_file", session_id=relative_session_id, filename=f"{pdf_name}/result.pdf", _external=True)

        # **Send Email if recipient email is provided**
        if recipient_email:
            if results_dict is None:
                email_context = {"Result": "Failed to calculate a model"}
            else:
                logging.info("Sending email!")
                if not results_dict["fail"]:
                    email_context = {
                        "luminosity": f"{luminosity:.1f}",
                        "teff": f"{teff:.1f}",
                        "mstar": f"{mstar:.1f}",
                        "rstar": "{:.2f}".format(results_dict["R_star"]),
                        "logg": "{:.2f}".format(results_dict["log_g"]),
                        "zstar": "{:.3e}".format(results_dict["Zmass"]),
                        "mass_loss_rate": "{:.3e}".format(results_dict["mdot"]),
                        "qbar": "{:.0f}".format(results_dict["Qbar"]),
                        "alpha": "{:.2f}".format(results_dict["alpha"]),
                        "q0": "{:.0f}".format(results_dict["Q0"]),
                        "vinf": "{:.2e}".format(results_dict["vinf"]),}
                    email_body = load_dyn_email('./mailing/mail_dyn.j2', email_context)
                else:
                    email_context = {"Result": results_dict["fail_reason"]}
                    email_body = load_dyn_email('./mailing/fail_template.j2', email_context)

            # Send the email using mailer.py
            subprocess.run([
                "python3", "./mailing/mailer.py",
                "--t", recipient_email,
                "--s", "LIME Computation Results",
                "--b", email_body])

        if expert_mode:
            with app.app_context():
                logger.info(f"Model calculation done PDF available at {pdf_url}")
                return {"message": "Computation complete", "download_url": pdf_url, **table_data}, 200
        else:
            return {"message": "Computation complete", **table_data}, 200
    
    except ValueError as e:
        logger.error(f"Invalid numerical input: {e}")
        with app.app_context():
            return {"error": f"Invalid numerical input: {str(e)}", **base_table_data}, 400
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        with app.app_context():
            return {"error": f"Unexpected error: {str(e)}", **base_table_data}, 500


@app.route("/task_status/<task_id>", methods=["GET"])
def get_processing_status(task_id):
    """Check how the calculation is doing"""
    task = celery.AsyncResult(task_id)
    if task.state == "SUCCESS":
        return jsonify({"status": task.state, "done": True, **task.result[0]}), task.result[1]
    else:
        return jsonify({"task_id": task.id, "status": task.state}), 202


@app.route('/tmp/<session_id>/<path:filename>')
def download_temp_file(session_id, filename):
    session_tmp_dir = os.path.join(ServerConfig.BASE_TMP_DIR, session_id)  
    file_path = os.path.abspath(os.path.join(session_tmp_dir, filename))  

    # Security check: Prevent directory traversal
    if not file_path.startswith(os.path.abspath(session_tmp_dir)):
        return jsonify({"error": "Invalid file path."}), 403

    # Debugging: Print expected path
    logger.debug(f"Looking for file at: {file_path}")

    if not os.path.exists(file_path):
        logger.error(f"File NOT FOUND: {file_path}")
        return jsonify({"error": f"File {filename} not found"}), 404

    response = send_file(file_path, mimetype='application/pdf')
    
    def cleanup():
        shutil.rmtree(session_tmp_dir, ignore_errors=True)
    
    threading.Timer(5, cleanup).start()  

    return response


def check_csv_input_file(df):
    """
    Checks if the required columns are present in the dataframe (generate from the csv file)
    Returns any missing columns
    """
    required_names = ["name", "teff", "luminosity", "mstar", "zscale"]
    column_names = df.columns.values.tolist()
    missing = []
    for column in required_names:
        if column not in column_names:
            missing.append(column)
    return missing


@app.route('/upload_csv', methods=['POST'])
def start_upload_csv():
    """This starts the processing of the csv file"""

    # First check if the file is uploaded and email address is supplied
    if 'file' not in request.files:
        logging.error("No CSV file provided for grid calculation")
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        logging.error("No file provided for grid calculation")
        return jsonify({"error": "No selected file"}), 400

    # Read in the file and check the number of lines.
    file = request.files["file"]
    file_data = file.read().decode("utf-8")
    num_rows = file_data.count("\n")

    if num_rows > 201:
        logging.error("Too long CSV file provided")
        return jsonify({
            "error": "Too many entries! Please reduce the number of stars to 200 or split the CSV into smaller parts."
        }), 400

    # Read in the data to check if all the columns are there
    df = pd.read_csv(io.BytesIO(file_data.encode()))
    missing_columns = check_csv_input_file(df)
    if len(missing_columns) > 0:
        logging.error(f"Missing columns in CSV file! Missing: {missing_columns}")
        return jsonify({"error": f"Missing columns! Missing: {missing_columns}"}), 400

    user_email = request.form.get("email", "").strip()
    if not user_email:
        logging.error("No email provided for CSV calculation results")
        return jsonify({"error": "Email is required"}), 400

    batch_tracker(num_rows - 1)
    # Schedule the process
    task = upload_csv.apply_async(args=[file_data, user_email])
    logging.info(f"Queueing grid calculation with ID: {task.id}")
    return jsonify({"task_id": task.id}), 200


@shared_task(ignore_result=True)
def upload_csv(file_data, user_email):
    """Handles CSV file upload and batch computation"""
    file_data = io.BytesIO(file_data.encode())
    df = pd.read_csv(file_data)

    logger.info(f"Starting batch calculation with size {len(df.index)}")

    # Create a single batch directory for all results
    base_dir = ServerConfig.BASE_TMP_DIR
    os.makedirs(base_dir, exist_ok=True)
    batch_output_dir = tempfile.mkdtemp(dir=base_dir)
    os.makedirs(batch_output_dir, exist_ok=True)

    logger.debug(f"Batch directory created: {batch_output_dir}")  # Debugging

    # **Return success response immediately**
    response_message = {
        "message": "CSV uploaded successfully. Your calculations will be processed and emailed to you shortly."
    }

    def process_batch():
        """Function to process the batch asynchronously"""
        try:
            default_abundances = {
                "H": 0.7374078505762753, "HE": 0.24924865007787272, "LI": 5.687053212055474e-11, "BE": 1.5816072816463046e-10,
                "B": 3.9638342804111373e-9, "C": 2.3649741118292409e-3, "N": 6.927752331287037e-4, "O": 5.7328054948662952e-3,
                "F": 5.0460905860356957e-7, "NE": 1.2565170515587217e-3, "NA": 2.9227131182144098e-6, "MG": 7.0785262928672096e-4,
                "AL": 5.5631575894102415e-5, "SI": 6.6484690760698845e-4, "P": 5.8243105278933166e-6, "S": 3.0923740022022601e-4,
                "CL": 8.2016309032581489e-6, "AR": 7.3407809644158897e-5, "K": 3.0647973602772301e-6, "CA": 6.4143590291084783e-5,
                "SC": 4.6455339921264288e-8, "TI": 3.1217731998425617e-6, "V": 3.1718648298183506e-7, "CR": 1.6604169480383736e-5,
                "MN": 1.0817329760692272e-5, "FE": 1.2919540666812507e-3, "CO": 4.2131387804051672e-6, "NI": 7.1254342166372973e-5,
                "CU": 7.2000506248032108e-7, "ZN": 1.7368347374506484e-6
            }

            results_csv_path = os.path.join(batch_output_dir, "results.csv")
            csv_header_written = False

            all_results = []
            all_mass_fractions = []

            for index, row in df.iterrows():
                pdf_name = str(row["name"])
                lum = float(row["luminosity"])
                teff = float(row["teff"])
                mstar = float(row["mstar"])
                zscale = float(row["zscale"])

                # Create subdirectory inside batch directory
                result_dir = os.path.join(batch_output_dir, pdf_name)
                os.makedirs(result_dir, exist_ok=True)

                abundances = {}
                total_metal_mass = 0
                for element, default_value in default_abundances.items():
                    if element in ["H", "HE"]:
                        abundances[element] = float(row[element]) if element in row and not pd.isna(row[element]) else default_value
                    else:
                        scaled_value = (float(row[element]) if element in row and not pd.isna(row[element]) else default_value) * zscale
                        abundances[element] = scaled_value
                        total_metal_mass += scaled_value

                zstar = calculate_metallicity_massb(abundances)

                #Keep He fixed and adjust H so the total remains 1
                helium_abundance = abundances["HE"]
                hydrogen_abundance = 1.0 - (total_metal_mass + helium_abundance)

                if hydrogen_abundance >= 0:
                    abundances["H"] = hydrogen_abundance
                    all_mass_fractions.append(abundances)
                else:
                    logger.info("Invalid input mass fractions, total mass (without hydrogen) > 1")
                    results_dict = dict(DUMMY_RESULTS)
                    results_dict["fail"] = True
                    results_dict["fail_reason"] = (f"Invalid input mass fractions,"
                                                   f" total mass without hydrogen > 1"
                                                   f" ({total_metal_mass + helium_abundance:.3f})")
                    all_results.append(results_dict)
                    all_mass_fractions.append(abundances)
                    continue  # Skip the rest of this iteration, no need to calculate further.

                # No need to make figures when running working through a batch calculation.
                does_plot = False

                # Start the main calculation
                results_dict = process_computation(lum, teff, mstar, zscale, zstar, helium_abundance, abundances,
                                                   batch_output_dir, does_plot)
                all_results.append(results_dict)

            for index, row in df.iterrows():
                results_dict = all_results[index]
                pdf_name = str(row["name"])
                result_dir = os.path.join(batch_output_dir, pdf_name)

                abundances_data = all_mass_fractions[index]

                with open(results_csv_path, mode='a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=["Name", "Luminosity", "Teff", "Mstar", "Rstar", "log g",
                                                           "Zstar", "Mass Loss Rate", "Gamma_e", "Qbar", "Alpha",
                                                           "Q0", "Vinf", "Success", "Remark", *abundances_data.keys()])
                    if not csv_header_written:
                        writer.writeheader()
                        csv_header_written = True
                    if not results_dict["fail"]:
                        writer.writerow({"Name": pdf_name,
                                         "Luminosity": row["luminosity"],
                                         "Teff": row["teff"],
                                         "Mstar": row["mstar"],
                                         "Rstar": "{:.2e}".format(results_dict["R_star"]),
                                         "log g": "{:.2e}".format(results_dict["log_g"]),
                                         "Zstar": "{:.3e}".format(results_dict["Zmass"]),
                                         "Mass Loss Rate": "{:.3e}".format(results_dict["mdot"]),
                                         "Gamma_e": "{:.3f}".format(results_dict["Gamma_e"]),
                                         "Qbar": "{:.2e}".format(results_dict["Qbar"]),
                                         "Alpha": "{:.2e}".format(results_dict["alpha"]),
                                         "Q0": "{:.2e}".format(results_dict["Q0"]),
                                         "Vinf": "{:.2e}".format(results_dict["vinf"]),
                                         "Success": str(not results_dict["fail"]),
                                         "Remark": results_dict["fail_reason"],
                                         **abundances_data})
                    else:
                        writer.writerow({"Name": pdf_name,
                                         "Luminosity": row["luminosity"],
                                         "Teff": row["teff"],
                                         "Mstar": row["mstar"],
                                         "Rstar": f"{np.nan}",
                                         "log g": f"{np.nan}",
                                         "Zstar": f"{np.nan}",
                                         "Mass Loss Rate": f"{np.nan}",
                                         "Gamma_e": f"{np.nan}",
                                         "Qbar": f"{np.nan}",
                                         "Alpha": f"{np.nan}",
                                         "Q0": f"{np.nan}",
                                         "Vinf": f"{np.nan}",
                                         "Success": "False",
                                         "Remark": results_dict["fail_reason"],
                                         **abundances_data})

            # Send email with ZIP
            body = load_email_body('./mailing/mail_template.j2')
            subprocess.run(["python3", "./mailing/mailer.py", "--t", user_email, "--s", "LIME Computation Results", "--b", body, "--a", results_csv_path])

        except Exception as e:
            logging.error(f"Ran into an unexpected error in batch processing: {e}")

        finally:
            # The batch directory is removed after processing
            logging.info("Cleaning batch process folder")
            shutil.rmtree(batch_output_dir, ignore_errors=True)

    # **Start computation in a new thread**
    batch_thread = threading.Thread(target=process_batch)
    batch_thread.start()

    logging.info("Finished batch process computation. ")
    return jsonify(response_message), 200
    #
    # except Exception as e:
    #     print(e)
    #     return jsonify({"error": f"Unexpected error in upload_csv: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
