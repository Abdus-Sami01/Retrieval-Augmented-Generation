from rag.rate_limit import RateLimiter, TokenBucket


def test_token_bucket_allows_up_to_capacity():
    bucket = TokenBucket(capacity=3, refill_per_second=0)
    assert bucket.try_consume() is True
    assert bucket.try_consume() is True
    assert bucket.try_consume() is True
    assert bucket.try_consume() is False


def test_token_bucket_refills_over_time():
    # Windows scheduler timer resolution is coarse, so use a wide margin: 200/sec
    # refill over a 100ms sleep should yield ~20 tokens even with significant slop.
    bucket = TokenBucket(capacity=1, refill_per_second=200)
    assert bucket.try_consume() is True
    assert bucket.try_consume() is False
    import time

    time.sleep(0.1)
    assert bucket.try_consume() is True


def test_rate_limiter_tracks_buckets_independently_per_key():
    limiter = RateLimiter(capacity=1, refill_per_second=0)
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is False
    assert limiter.allow("client-b") is True


def test_rate_limiter_denies_after_capacity_exhausted():
    limiter = RateLimiter(capacity=2, refill_per_second=0)
    assert limiter.allow("x") is True
    assert limiter.allow("x") is True
    assert limiter.allow("x") is False
