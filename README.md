# Insider Trading Predictor

This repository contains a machine learning model that predicts which insider trades are likely to result in stock price increases.

## Files

- `predict_today_trades_github.py` - Main prediction script (modified for GitHub Actions)
- `insider_trading_model.joblib` - Trained XGBoost model with optimal threshold
- `requirements.txt` - Python dependencies
- `.github/workflows/daily-predictions.yml` - GitHub Actions workflow

## Setup Instructions

### 1. Create GitHub Repository
1. Go to [GitHub.com](https://github.com) and create a new repository
2. Name it `insider-trading-predictor` (or your preferred name)

### 2. Upload Files
Upload these files to your repository:
- `predict_today_trades_github.py` (rename to `predict_today_trades.py`)
- `insider_trading_model.joblib`
- `requirements.txt`
- `.github/workflows/daily-predictions.yml`

### 3. Configure Email Notifications (Optional)
If you want email notifications:

1. Go to your repository Settings → Secrets and variables → Actions
2. Add these repository secrets:
   - `MAIL_USERNAME`: Your Gmail address
   - `MAIL_PASSWORD`: Your Gmail app password (not regular password)
   - `MAIL_TO`: Email address to receive notifications

### 4. Test the Workflow
1. Go to Actions tab in your repository
2. Click "Daily Insider Trading Predictions"
3. Click "Run workflow" to test it manually

## How It Works

The workflow runs daily at 9:00 AM UTC and:
1. Fetches today's insider trades from OpenInsider
2. Downloads current stock data and fundamentals
3. Processes the data using your trained model
4. Predicts which trades are likely to result in price increases
5. Saves results as CSV artifacts
6. Sends email notification (if configured)

## Customization

- **Change schedule**: Edit the `cron` line in `.github/workflows/daily-predictions.yml`
- **Change timezone**: The current schedule is 9:00 AM UTC. Adjust as needed
- **Manual runs**: You can trigger the workflow manually anytime

## Model Details

- **Algorithm**: XGBoost with optimized hyperparameters
- **Threshold**: 0.4 (optimized for balanced precision/recall)
- **Features**: Technical indicators, fundamental data, insider role, trade characteristics
- **Performance**: ~82% accuracy on test set

## Troubleshooting

- Check the Actions tab for detailed logs
- Ensure all dependencies are in `requirements.txt`
- Verify the model file is uploaded correctly
- Check email credentials if notifications aren't working 