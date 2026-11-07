#pragma once

#include <array>
#include <boost/unordered/unordered_flat_map.hpp>
#include <cstdint>
#include <span>
#include <valarray>
#include <vector>

#include "averager.h"
#include "utils.h"

struct InfosetMetadata {
  uint32_t legal_actions;
  uint32_t infoset_id;

  bool operator==(const InfosetMetadata &other) const {
    return legal_actions == other.legal_actions &&
           infoset_id == other.infoset_id;
  }
};

// Maps from infoset to legal action mask
using InfosetMap = boost::unordered_flat_map<uint64_t, InfosetMetadata>;

struct Treeplex {
  InfosetMap infosets;
  std::vector<uint64_t> infoset_keys;
  std::vector<uint32_t> legal_actions;
  std::vector<uint32_t> parent_index;

  uint32_t num_infosets() const { return infoset_keys.size(); }
  bool is_valid_vector(ConstRealBuf buf) const;
  bool is_valid_strategy(ConstRealBuf buf) const;
  void set_uniform(RealBuf buf) const;
  void bh_to_sf(RealBuf buf) const;
  void sf_to_bh(RealBuf buf) const;
  Real br(RealBuf grad, RealBuf strat = std::span<Real>()) const;
  void regret_to_bh(RealBuf buf) const;
};

struct EvExpl {
  Real ev0;
  // gradient of utility wrt the player strategies
  PerPlayer<std::valarray<Real>> gradient;
  // expl[0] is how exploitable player 0 is by a best-responding player 1
  PerPlayer<Real> expl;
  // best_response[0] is the best response to player 1's strategy
  PerPlayer<std::valarray<Real>> best_response;
};

template <typename T> struct Traverser {
  PerPlayer<std::shared_ptr<Treeplex>> treeplex;
  PerPlayer<std::valarray<Real>> gradients;
  Traverser();

  void compute_gradients(const PerPlayer<ConstRealBuf> strategies);
  EvExpl ev_and_exploitability(const PerPlayer<ConstRealBuf> strategies);
  Averager new_averager(const uint8_t player, const AveragingStrategy avg);
  void compute_openspiel_infostate(const uint8_t player, int64_t i, std::span<bool> buf) const;
  void compute_openspiel_infostates(const uint8_t player, std::span<bool> buf) const;

  // --- Sequence-form payoff matrix A (utility of player 0) -----------------
  // Sequences are indexed as 9 * infoset_row + action (same layout as the
  // flattened (NUM_INFOS, 9) strategy / gradient arrays). Then
  //   ev0 = x^T A y,  gradient[0] = A y,  gradient[1] = -A^T x.
  // The game tree is split into 81 subtrees: subtree i starts after player 0
  // plays cell i % 9 and player 1 plays cell i / 9 (as in compute_gradients).
  // count_utility_nonzeros returns the number of decisive (non-tie) leaves of
  // each selected subtree (0 for unselected ones); fill_utility_matrix writes
  // the COO triplets of the selected subtrees, in subtree order, into buffers
  // of size sum(counts). Payoffs are +1 (player 0 wins) / -1 (player 1 wins).
  static constexpr int NUM_SUBTREES = 81;
  using SubtreeMask = std::array<bool, NUM_SUBTREES>;
  using SubtreeCounts = std::array<uint64_t, NUM_SUBTREES>;
  SubtreeCounts count_utility_nonzeros(const SubtreeMask &selected) const;
  void fill_utility_matrix(const SubtreeMask &selected,
                           const SubtreeCounts &counts,
                           std::span<uint32_t> seq0, std::span<uint32_t> seq1,
                           std::span<int8_t> u0) const;

private:
  PerPlayer<std::array<std::valarray<Real>, 9>> bufs_;
  PerPlayer<std::valarray<Real>> sf_strategies_;

  void compute_sf_strategies_(const PerPlayer<ConstRealBuf> strategies);
};
