#!/usr/bin/env python3
"""
CLI for signal generation and labeling system
"""
import click
from loguru import logger
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_pipeline import ModelPipeline


@click.group()
def cli():
    """Personal Quant Desk - Signal Generation & Labeling System"""
    pass


@cli.command()
@click.option('--symbol', required=True, help='Symbol to process (e.g., SPY)')
@click.option('--action',
              type=click.Choice(['label', 'train', 'signal', 'backtest', 'all']),
              default='all',
              help='Action to perform')
@click.option('--use-meta', is_flag=True, default=True, help='Use meta-labeling')
@click.option('--config', default='models/config/model_config.yaml', help='Path to config file')
def process(symbol: str, action: str, use_meta: bool, config: str):
    """Process model pipeline for a symbol"""
    logger.info(f"Processing {symbol} with action={action}, use_meta={use_meta}")

    try:
        # Initialize pipeline
        pipeline = ModelPipeline(config_path=config)

        # Execute requested actions
        if action in ['label', 'all']:
            logger.info(f"[Step 1/4] Generating labels for {symbol}")
            labels = pipeline.run_triple_barrier_labeling(symbol)
            click.echo(f"✓ Generated {len(labels)} labels for {symbol}")
            click.echo(f"  Label distribution: {labels['label'].value_counts().to_dict()}")

        if action in ['train', 'all']:
            logger.info(f"[Step 2/4] Training models for {symbol}")

            # Train primary model
            click.echo(f"\n→ Training primary model...")
            model = pipeline.train_primary_model(symbol)
            click.echo(f"✓ Primary model trained")

            # Train meta-model if requested
            if use_meta:
                click.echo(f"\n→ Training meta-model...")
                meta_model = pipeline.train_meta_model(symbol)
                click.echo(f"✓ Meta-model trained")

        if action in ['signal', 'all']:
            logger.info(f"[Step 3/4] Generating signals for {symbol}")
            signals = pipeline.generate_signals(symbol, use_meta=use_meta)
            click.echo(f"\n✓ Generated {len(signals)} signals for {symbol}")
            click.echo(f"  Signal distribution:")
            click.echo(f"    Long signals:  {(signals['signal'] > 0).sum()}")
            click.echo(f"    Short signals: {(signals['signal'] < 0).sum()}")
            click.echo(f"    Neutral:       {(signals['signal'] == 0).sum()}")

            # Save signals
            pipeline.save_signals(symbol)
            click.echo(f"✓ Signals saved to models/signals/{symbol}_signals.parquet")

        if action in ['backtest', 'all']:
            logger.info(f"[Step 4/4] Backtesting signals for {symbol}")
            metrics = pipeline.backtest_signals(symbol)

            click.echo(f"\n{'='*60}")
            click.echo(f"BACKTEST RESULTS for {symbol}")
            click.echo(f"{'='*60}")
            click.echo(f"Total signals:    {metrics['total_signals']}")
            click.echo(f"  Long signals:   {metrics['long_signals']}")
            click.echo(f"  Short signals:  {metrics['short_signals']}")
            click.echo(f"Hit rate:         {metrics['hit_rate']:.2%}")
            click.echo(f"Avg return:       {metrics['avg_return']:.4f}")
            click.echo(f"Total return:     {metrics['total_return']:.4f}")
            click.echo(f"Sharpe ratio:     {metrics['sharpe_ratio']:.4f}")
            click.echo(f"Max drawdown:     {metrics['max_drawdown']:.4f}")
            click.echo(f"{'='*60}\n")

        click.echo(f"\n✓ Pipeline completed successfully for {symbol}")

    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--symbols', required=True, help='Comma-separated list of symbols')
@click.option('--use-meta', is_flag=True, default=True, help='Use meta-labeling')
@click.option('--config', default='models/config/model_config.yaml', help='Path to config file')
def batch(symbols: str, use_meta: bool, config: str):
    """Process multiple symbols in batch"""
    symbol_list = [s.strip() for s in symbols.split(',')]

    click.echo(f"Processing {len(symbol_list)} symbols: {', '.join(symbol_list)}")

    pipeline = ModelPipeline(config_path=config)
    results = {}

    for i, symbol in enumerate(symbol_list, 1):
        click.echo(f"\n[{i}/{len(symbol_list)}] Processing {symbol}...")

        try:
            # Run full pipeline
            labels = pipeline.run_triple_barrier_labeling(symbol)
            model = pipeline.train_primary_model(symbol)

            if use_meta:
                meta_model = pipeline.train_meta_model(symbol)

            signals = pipeline.generate_signals(symbol, use_meta=use_meta)
            metrics = pipeline.backtest_signals(symbol)

            results[symbol] = {
                'status': 'success',
                'labels': len(labels),
                'signals': len(signals),
                'hit_rate': metrics['hit_rate'],
                'sharpe': metrics['sharpe_ratio']
            }

            click.echo(f"  ✓ {symbol}: Hit rate={metrics['hit_rate']:.2%}, Sharpe={metrics['sharpe_ratio']:.2f}")

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            results[symbol] = {'status': 'error', 'error': str(e)}
            click.echo(f"  ✗ {symbol}: {e}")

    # Summary
    click.echo(f"\n{'='*60}")
    click.echo("BATCH PROCESSING SUMMARY")
    click.echo(f"{'='*60}")

    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    click.echo(f"Successful: {success_count}/{len(symbol_list)}")

    if success_count > 0:
        click.echo("\nResults:")
        for symbol, result in results.items():
            if result['status'] == 'success':
                click.echo(f"  {symbol}: Hit={result['hit_rate']:.2%}, Sharpe={result['sharpe']:.2f}")

    click.echo(f"{'='*60}\n")


@cli.command()
@click.option('--symbol', required=True, help='Symbol to analyze')
@click.option('--config', default='models/config/model_config.yaml', help='Path to config file')
def analyze(symbol: str, config: str):
    """Analyze feature importance for a symbol"""
    click.echo(f"Analyzing feature importance for {symbol}...")

    try:
        pipeline = ModelPipeline(config_path=config)

        # Ensure model is trained
        if f'{symbol}_primary' not in pipeline.models:
            click.echo("Training model first...")
            pipeline.train_primary_model(symbol)

        # Get feature importance
        model_data = pipeline.models[f'{symbol}_primary']
        importances = model_data['importances']

        click.echo(f"\n{'='*60}")
        click.echo(f"FEATURE IMPORTANCE for {symbol}")
        click.echo(f"{'='*60}\n")

        # Show top features from each method
        for method in ['mdi', 'mda', 'combined']:
            if method in importances:
                click.echo(f"\nTop 10 features ({method.upper()}):")
                top_features = importances[method].head(10)
                for i, (feature, score) in enumerate(top_features.items(), 1):
                    click.echo(f"  {i:2d}. {feature:30s} {score:.6f}")

        click.echo(f"\n{'='*60}\n")

    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--symbol', required=True, help='Symbol to load model for')
@click.option('--model-type', type=click.Choice(['primary', 'meta']), default='primary', help='Model type')
def load_model(symbol: str, model_type: str):
    """Load and inspect a saved model"""
    try:
        pipeline = ModelPipeline()
        model = pipeline.load_model(symbol, model_type)

        click.echo(f"✓ Loaded {model_type} model for {symbol}")
        click.echo(f"  Model type: {type(model).__name__}")

        if hasattr(model, 'feature_names_in_'):
            click.echo(f"  Number of features: {len(model.feature_names_in_)}")

    except Exception as e:
        click.echo(f"✗ Error loading model: {e}", err=True)
        sys.exit(1)


@cli.command()
def info():
    """Show system information and available symbols"""
    click.echo("Personal Quant Desk - Signal Generation System")
    click.echo("Based on López de Prado's 'Advances in Financial ML'\n")

    click.echo("Available commands:")
    click.echo("  process   - Process a single symbol")
    click.echo("  batch     - Process multiple symbols")
    click.echo("  analyze   - Analyze feature importance")
    click.echo("  load-model- Load and inspect a model")
    click.echo("  info      - Show this information")

    click.echo("\nExample usage:")
    click.echo("  python models/main.py process --symbol SPY --action all")
    click.echo("  python models/main.py batch --symbols SPY,QQQ,IWM")
    click.echo("  python models/main.py analyze --symbol SPY")


if __name__ == '__main__':
    cli()
