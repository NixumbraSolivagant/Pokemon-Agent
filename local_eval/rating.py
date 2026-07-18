from __future__ import annotations

from dataclasses import dataclass

import trueskill

from .models import EvalConfig, GameResult


@dataclass(slots=True)
class RatingState:
    rating: trueskill.Rating

    @property
    def mu(self) -> float:
        return float(self.rating.mu)

    @property
    def sigma(self) -> float:
        return float(self.rating.sigma)

    @property
    def kaggle_score_estimate(self) -> float:
        return float(self.rating.mu - 3.0 * self.rating.sigma)


class KaggleStyleRating:
    def __init__(self, config: EvalConfig):
        self.env = trueskill.TrueSkill(
            mu=config.trueskill_mu,
            sigma=config.trueskill_sigma,
            beta=config.trueskill_beta,
            tau=config.trueskill_tau,
            draw_probability=config.trueskill_draw_probability,
        )

    def new_rating(self) -> RatingState:
        return RatingState(self.env.create_rating())

    def update_game(self, p0: RatingState, p1: RatingState, result: GameResult) -> tuple[RatingState, RatingState]:
        if result.outcome == "DRAW":
            ranks = [0, 0]
        elif result.winner == result.p0:
            ranks = [0, 1]
        elif result.winner == result.p1:
            ranks = [1, 0]
        else:
            return p0, p1
        (new_p0,), (new_p1,) = self.env.rate([(p0.rating,), (p1.rating,)], ranks=ranks)
        return RatingState(new_p0), RatingState(new_p1)
