# Smart Surveillance System - Backend

This document provides instructions for setting up and running the backend of the Smart Surveillance System.

## 1. Installation

### Prerequisites

- Python 3.9+
- MySQL
- An environment that supports `asyncio`

### Setup

1.  **Navigate to the Backend Directory**

    ```bash
    cd Backend
    ```

2.  **Create a Virtual Environment**

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**

    Create a `.env` file in the `Backend` directory and add the following variables. Replace the placeholder values with your actual database credentials.

    ```env
    DATABASE_HOST=localhost
    DATABASE_USER=your_db_user
    DATABASE_PASSWORD=your_db_password
    DATABASE_NAME=surveillance_db
    SECRET_KEY=a_very_secret_key
    ```

5.  **Initialize the Database**

    Run the following script to create the database, tables, and default users:

    ```bash
    python init_mysql_db.py
    ```

    This will create an `admin` user (password: `admin123`) and a `viewer` user (password: `viewer123`).

## 2. Model Training

Before running the application, you need to train the crime detection model.

1.  **Download the Dataset**: Download the [UCF-Crime](https://www.crcv.ucf.edu/data/ucf-crime-dataset/) dataset.

2.  **Update the Training Script**: Open `train_crime_model.py` and update the `dataset_path` variable to point to the location of your dataset.

3.  **Run the Training Script**:

    ```bash
    python train_crime_model.py
    ```

    This will train the model and save it to the `models/` directory.

## 3. Running the Application

Once the setup is complete, you can start the backend server:

```bash
python main.py
```

The API will be available at `http://localhost:8000`.

## 4. API Documentation

Interactive API documentation is available at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 5. Next Steps and acks

This backend is now much more stable and reliable. For future development, consider the following improvements:

- **Use Celery**: For more demanding background tasks, consider integrating Celery for a more robust and scalable solution.
- **Frontend Enhancements**: Improve the frontend to provide real-time feedback on file uploads and processing status.
- **Deployment**: Containerize the application with Docker for easier deployment.
