# ThyroCheck Python Backend

Backend service for ThyroCheck thyroid health assessment application built with Flask.

## Features

- **Email Service**: Send personalized health assessment results via email
- **PDF Reports**: Generate professional PDF reports using ReportLab
- **Security**: Rate limiting and input validation
- **CORS Support**: Configured for frontend communication
- **Professional Reports**: HTML-formatted medical reports

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Configuration**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` file with your email credentials:
   ```
   EMAIL_USER=your-email@gmail.com
   EMAIL_PASS=your-app-password
   SMTP_SERVER=smtp.gmail.com
   SMTP_PORT=587
   PORT=5000
   FLASK_ENV=development
   ```

3. **Email Setup (Gmail)**
   - Enable 2-factor authentication on your Gmail account
   - Generate an App Password: https://support.google.com/accounts/answer/185833
   - Use the App Password (not your regular password) in the `.env` file

4. **Start Server**
   ```bash
   python app.py
   ```

## API Endpoints

### POST /api/send-results
Send health assessment results via email.

**Request Body:**
```json
{
  "email": "user@example.com",
  "name": "John Doe",
  "results": {
    "score": 8,
    "date": "11/1/2025",
    "time": "2:30 PM",
    "symptomDetails": ["fatigue", "weight_changes"]
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Health assessment results sent to your email successfully!"
}
```

### POST /api/download-report
Generate and download PDF report.

**Request Body:**
```json
{
  "name": "John Doe",
  "results": {
    "score": 8,
    "date": "11/1/2025",
    "time": "2:30 PM",
    "symptomDetails": ["fatigue", "weight_changes"]
  }
}
```

### GET /api/health
Health check endpoint.

## Security Features

- **Rate Limiting**: 10 requests per 15 minutes per IP
- **Input Validation**: Email format and required fields
- **CORS**: Configured for frontend communication

## Email Template

The service generates professional HTML emails with:
- Personalized health assessment results
- Medical score and interpretation
- Symptom analysis
- Tailored recommendations
- Professional formatting

## PDF Reports

Professional PDF reports include:
- Patient information and assessment date
- Health score and status
- Detailed symptom analysis
- Personalized recommendations
- Medical disclaimer

## Development

- Uses Flask web framework
- SMTP for email functionality
- ReportLab for PDF generation
- Environment-based configuration
- Error logging and handling

## Deployment

1. Set up environment variables on your hosting platform
2. Ensure email service credentials are configured
3. Install dependencies: `pip install -r requirements.txt`
4. Start the server: `python app.py`
5. The frontend will connect to `http://localhost:5000` (or your configured port)

## Troubleshooting

**Email not sending:**
- Verify Gmail App Password is correct
- Check if 2FA is enabled on Gmail account
- Ensure firewall allows SMTP connections

**CORS errors:**
- Make sure the backend is running on the correct port
- Check CORS configuration in the Flask app

**Rate limiting:**
- Wait 15 minutes if you hit the rate limit
- Contact administrator if persistent issues

**PDF generation issues:**
- Ensure ReportLab is properly installed
- Check file permissions for temporary file creation