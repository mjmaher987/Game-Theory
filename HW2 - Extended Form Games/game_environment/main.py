from abc import abstractmethod
from dataclasses import dataclass, field
from random import choices
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


Offer = Tuple[int, int]


class Strategy:
    @abstractmethod
    def generate_offer(self, **kwargs) -> Offer:
        ...

    @abstractmethod
    def receive_offer(self, **kwargs) -> bool:
        ...

    def start_round(self):
        ...

    @property
    def name(self):
        return self.__class__.__name__


@dataclass
class Player:
    idx: int
    strategy: Strategy
    asset: int = field(default=1)

    def generate_offer(self, *args, **kwargs) -> Offer:
        return self.strategy.generate_offer(*args, your_idx=self.idx, **kwargs)

    def receive_offer(self, *args, **kwargs) -> bool:
        return self.strategy.receive_offer(*args, your_idx=self.idx, **kwargs)

    def start_round(self):
        self.strategy.start_round()

    @property
    def name(self):
        return self.strategy.__class__.__name__


@dataclass
class GameConfiguration:
    strategies: List[Strategy]  # Strategy for each player
    offers: int = field(default=1)  # Number of possible offers in each game
    player2_offers: bool = field(default=False)  # If False, only player1 places offers
    value: int = field(default=100)
    show_history: bool = field(
        default=False
    )  # If True, players will have access to past offers and their results
    games: int = field(default=1)  # Number of games held randomly between players


class GameRunner:
    """
    Instantiates a game between two players with using a configuration setting
    parameters for the game.
    """

    config: GameConfiguration
    players: List[Player]
    round_results: Dict[str, List[int]]

    def __init__(self, config: GameConfiguration):
        self.config = config
        self.players = [
            Player(idx, strategy) for idx, strategy in enumerate(self.config.strategies)
        ]
        self.round_results = {}
        self.games = []

    def play(self, players: Tuple[Player, Player]) -> Tuple[int, int]:
        """
        Runs the game.

        Returns A tuple representing the final reward of player1 and player2 respectively
        """

        def validate_offer(offer: Offer) -> bool:
            return (
                offer[0] >= 0
                and offer[1] >= 0
                and offer[0] + offer[1] == self.config.value
            )

        for player in players:
            player.start_round()
        offers = []
        for turn in range(self.config.offers):
            giver_idx = (turn % 2) * self.config.player2_offers
            offer = players[giver_idx].generate_offer(
                value=self.config.value,
                previous_games=self.games,
                receiver_idx=players[1 - giver_idx].idx,
            )
            offers.append(offer)
            assert validate_offer(offer)
            accepted = players[1 - giver_idx].receive_offer(
                offer=offer,
                previous_games=self.games,
                giver_idx=players[giver_idx].idx,
            )
            self.record_game(
                players[giver_idx], players[1 - giver_idx], offers, accepted
            )
            if accepted:
                players[giver_idx].asset += offer[0]
                players[1 - giver_idx].asset += offer[1]
                break

    def run(self):
        for i in range(self.config.games):
            players = choices(self.players, k=2)
            self.play(players)
            self.store_assets()

    def store_assets(self):
        current_assets = self.get_assets()
        s = sum(current_assets.values())
        for key in current_assets:
            current_assets[key] /= s

        for key in current_assets:
            if key not in self.round_results:
                self.round_results[key] = []
            self.round_results[key].append(current_assets[key])

    def get_assets(self):
        current_assets = {}
        for player in self.players:
            if player.name not in current_assets:
                current_assets[player.name] = 0
            current_assets[player.name] += player.asset
        return current_assets

    def record_game(
        self,
        giver: Player,
        receiver: Player,
        offers: List[Tuple[int, int]],
        accepted: bool,
    ):
        self.games.append(
            {
                "initial giver": giver.idx,
                "initial receiver": receiver.idx,
                "offers": offers,
                "accepted": accepted,
            }
        )

    def plot(self):
        x = np.arange(0, self.config.games)
        y = np.vstack(list(self.round_results.values()))

        fig, ax = plt.subplots()
        ax.stackplot(
            x,
            y,
            labels=self.round_results.keys(),
        )
        ax.set(ylim=(0, 1), xlabel="#round", ylabel="share of assets")
        ax.legend()

        plt.show()
