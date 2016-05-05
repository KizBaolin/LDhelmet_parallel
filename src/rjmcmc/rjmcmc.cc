// Copyright (C) 2012  Andrew H. Chan, Paul A. Jenkins, Yun S. Song
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// <http://www.gnu.org/licenses/>
//
// email: andrewhc@eecs.berkeley.edu

#include "rjmcmc/rjmcmc.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <vector>
#include <omp.h>
#include <inttypes.h>
#include <ctime>

#include "common/binary_search.h"
#include "common/load_data.h"
#include "common/log_lk_computer.h"
#include "common/seq_process.h"
#include "rjmcmc/acceptance_log.h"
#include "rjmcmc/post_rho_map.h"
#include "rjmcmc/priors.h"
#include "rjmcmc/proposals.h"
#include "rjmcmc/ran_num_gen.h"

Rjmcmc::Rjmcmc(
    RanNumGen::SeedType seed,
    std::vector<uint64_t> const &snp_pos,
    LogLkComputer const &log_lk_computer,
    uint32_t num_iter,
    uint32_t burn_in,
    ProposalDist const &proposal_dist,
    AcceptanceRatio const &acceptance_ratio,
    double max_lk_start,
    double max_lk_end,
    double max_lk_resolution,
    uint32_t window_size,
    uint64_t stats_thin,
    uint64_t partition_start,
    uint64_t partition_end,
    std::vector<std::vector<std::pair<uint64_t, double> > > &sample_store,
    // parallel tempering options
    uint32_t num_mcmc_chains,
    double temp_const,
    uint32_t num_iter_swap)
    : rng_(seed),
      uniform_gen_(rng_.GetUniformGen(0.0, 1.0)),
      rate_change_gen_(rng_.GetUniformGen(-0.5, 0.5)),
      num_iter_(num_iter),
      burn_in_(burn_in),
      window_size_(window_size),
      stats_thin_(stats_thin),
      acceptance_ratio_(acceptance_ratio),
      proposals_(proposal_dist),
      snp_pos_(snp_pos),
      log_lk_computer_(log_lk_computer),
      iteration_id_(0),
      post_rho_map_(snp_pos_.size()),
      partition_start_(partition_start),
      partition_end_(partition_end),
      sample_store_(sample_store),
      // parallel tempering options
      num_mcmc_chains_(num_mcmc_chains),
      temp_const_(temp_const),
      num_iter_swap_(num_iter_swap) {
  // SNP position file needs at least 2 positions.
  if (snp_pos_.size() < 2) {
    // fprintf(stderr, "Need at least 2 SNPs.\n");
    std::exit(1);
  }

  if (window_size_ < 2) {
    // fprintf(stderr, "Window size must be at least 2.\n");
    std::exit(1);
  }

  // SNP positions must be in increasing order
  for (size_t snp_id = 0; snp_id < snp_pos_.size() - 1; ++snp_id) {
    if (snp_pos_[snp_id] >= snp_pos_[snp_id + 1]) {
      // fprintf(stderr,"SNP positions must be in increasing order.\n");
      std::exit(1);
    }
  }

  // Total physical distance.
  uint32_t tot_length = snp_pos_.back() - snp_pos_.front();
  assert(tot_length > 0);

  double max_lk_rate =
    FullLogLkComputer(snp_pos_, log_lk_computer_)
      .ComputeMaxLkConstantRho(max_lk_start, max_lk_end, max_lk_resolution);
  if (max_lk_rate < 0.0) {
    // fprintf(stderr,"Error: Maximum likelihood estimate of rho is negative. There is probably a bug in the code.\n");
    std::exit(1);
  }

  double init_rate = max_lk_rate;
  assert(init_rate >= 0.0);

  for (size_t snp_id = 0; snp_id < snp_pos_.size() - 1; ++snp_id) {
    change_points_.push_back(ChangePoint(snp_id, init_rate));
  }

  // Last change point; rate is not needed.
  change_points_.push_back(ChangePoint(snp_pos_.size() - 1, -1.0));

  // printf("Print change_points_ initial capacity: %d", change_points_.capacity());
  // printf("Print change_points_ initial size: %d", change_points_.size());

  proposed_cum_rho_map_ = std::vector<double>(snp_pos_.size(), 0.0);
  UpdateCumRhoMap(&proposed_cum_rho_map_, 0);
  cum_rho_map_ = proposed_cum_rho_map_;

  cur_log_lk = 0.0;
  log_lk_map_ = std::vector<double>(snp_pos_.size() * window_size_);
  proposed_log_lk_map_ = std::vector<double>(snp_pos_.size()*window_size_);
  cur_log_lk = ProposeLogLk(0, snp_pos_.size());
  UpdateLogLkMap(0, snp_pos_.size());

  rechosen_snp_ = 0;

  burn_in_p_ = false;  // Needed for acceptance logger.
}

void Rjmcmc::run() {

  burn_in_p_ = true;  // Needed for acceptance logger.
  for (uint32_t iteration = 0; iteration < burn_in_; ++iteration) {
    Update();
    assert(change_points_.size() >= 2);
  }

  burn_in_p_ = false;  // Needed for acceptance logger.

  std::clock_t start = std::clock();
  //*****************************************Parallel Tempering stuff******************************//
  int nthreads = num_mcmc_chains_;
  // printf("Number of threads = %d\n", nthreads);
  omp_set_num_threads(nthreads);

  std::vector<double> T;
  std::vector<double> lpost;
  T.resize(nthreads);
  lpost.resize(nthreads);
  for (int i = 0; i < nthreads; ++i) {
    T[i]=pow(1.0/temp_const_, i);
    // printf("T array: %f\n", T[i]);
  }

  std::vector<int> rejection_count(nthreads,0);
  std::vector<int> swap_count(nthreads,0);

  int rank, rank_partner;

  //inititalize class data members as local variables
  double cur_log_lk_local;
  std::vector<double> cum_rho_map_local;
  std::vector<double> proposed_cum_rho_map_local;
  std::vector<ChangePoint> change_points_local;
  LogLkMap log_lk_map_local;
  LogLkMap proposed_log_lk_map_local;

  // printf("Print cur_log_lk Before pragma %f \n", cur_log_lk);

  /* Fork a team of threads giving them their own copies of variables */
  #pragma omp parallel private(rank, rank_partner, cur_log_lk_local, cum_rho_map_local, proposed_cum_rho_map_local, change_points_local, log_lk_map_local, proposed_log_lk_map_local) shared(lpost)
  {
    rank = omp_get_thread_num();
    // printf("Hello World from thread = %d\n", rank);
    PostRhoMap post_rho_map_local(snp_pos_.size());

    cur_log_lk_local = T[rank]*cur_log_lk;
    change_points_local = change_points_;
    cum_rho_map_local = cum_rho_map_;
    proposed_cum_rho_map_local = proposed_cum_rho_map_;
    log_lk_map_local = log_lk_map_;
    proposed_log_lk_map_local = proposed_log_lk_map_;

    for (uint32_t iteration = 0; iteration < num_iter_; ++iteration) 
    {
      if (T[rank] == 1) {
        RecordSample_Tempering(&change_points_local);
      }

      lpost[rank] = cur_log_lk_local;
      Update_Tempering(&cur_log_lk_local, &cum_rho_map_local, &proposed_cum_rho_map_local, &change_points_local, &log_lk_map_local, &proposed_log_lk_map_local, &post_rho_map_local, T[rank], &rejection_count, rank);
      assert(change_points_local.size() >= 2);

      // printf("Print cur_log_lk_local after pragma %f \n", lpost[rank]);
      // printf("Print cur_log_lk after pragma %f \n", cur_log_lk);

      post_rho_map_local.Update(snp_pos_, cum_rho_map_);
    
      if(num_iter_ % num_iter_swap_ == 0)
      {
        #pragma omp barrier      //Synchronise Threads
        /******************************************Inter-Thread Parallel Tempering**************/
        #pragma omp critical     
        {
          double random_uniform_prob = uniform_gen_();
          // printf("random_uniform_prob =  %f \n", random_uniform_prob);

          rank_partner = rank + 1;
          if(rank_partner < nthreads) {
            double lalpha = (T[rank]-T[rank_partner])*(lpost[rank_partner]-lpost[rank]);
            // printf("lalpha =  %f \n", lalpha);
            double random_log_uniform_prob = std::log(uniform_gen_());
            // printf("random_log_uniform_prob =  %f \n", random_log_uniform_prob);
            if(random_log_uniform_prob < lalpha){
              std::swap(T[rank],T[rank_partner]);
              // printf("Swapped %d with %d \n", rank, rank_partner);
              swap_count[rank] +=1;
            }
          }         
        }
      }
    }


    //Reset class data members
    if(T[rank] == 1){
      cur_log_lk = cur_log_lk_local;
      cum_rho_map_ = cum_rho_map_local;
      proposed_cum_rho_map_ = proposed_cum_rho_map_local;
      change_points_ = change_points_local;
      log_lk_map_ = log_lk_map_local;
      proposed_log_lk_map_ = proposed_log_lk_map_local;
      post_rho_map_ = post_rho_map_local;
    }

  }  /* All threads join master thread and disband */
    int sum = 0;

    for (int i = 0; i < nthreads; ++i) {
        printf("Number of rejections for thread %d: %d\n", i, rejection_count[i]);
        sum += rejection_count[i];
    }
    printf("Sum of rejection_count: %d\n", sum);
    printf("Number of iterations: %du\n", num_iter_);

  double duration = (std::clock() - start)/(double) CLOCKS_PER_SEC;
  printf("Duration: %f\n", duration);
}

/************************************************************Parallel Tempering Methods BEGIN**********************************/
void Rjmcmc::Update_Tempering(double *cur_log_lk_local, std::vector<double> *cum_rho_map_local, std::vector<double> *proposed_cum_rho_map_local, 
  std::vector<ChangePoint> *change_points_local, LogLkMap *log_lk_map_local, LogLkMap *proposed_log_lk_map_local, PostRhoMap *post_rho_map_local, double thread_temp, std::vector<int> *rejection_count, int rank) {
  double uniform_variate = uniform_gen_();
    // printf("change points local pointer: %d\n", change_points_local->size());
    // printf("change points pointer: %d\n", change_points_.size());
    // printf("change points local pointer capacity: %d\n", change_points_local->capacity());
    // printf("change points pointer capacity: %d\n", change_points_.capacity());
    // printf("change points local pointer maxsize: %zu\n", change_points_local->max_size());
    // printf("change points pointer maxsize: %zu\n", change_points_.max_size());
  if (uniform_variate < proposals_.cum_.change) {
    // printf("Before Update_Tempering, Change\n");
    PerformChange_Tempering(cur_log_lk_local, cum_rho_map_local, proposed_cum_rho_map_local, change_points_local, log_lk_map_local, proposed_log_lk_map_local, post_rho_map_local, thread_temp, rejection_count, rank);
  } else if (uniform_variate < proposals_.cum_.extend) {
    // printf("Before Update_Tempering, Extend\n");
    PerformExtend_Tempering(cur_log_lk_local, cum_rho_map_local, proposed_cum_rho_map_local, change_points_local, log_lk_map_local, proposed_log_lk_map_local, post_rho_map_local, thread_temp, rejection_count, rank);
  } else if (uniform_variate < proposals_.cum_.split) {
    // printf("Before Update_Tempering, Split\n");
    PerformSplit_Tempering(cur_log_lk_local, cum_rho_map_local, proposed_cum_rho_map_local, change_points_local, log_lk_map_local, proposed_log_lk_map_local, post_rho_map_local, thread_temp, rejection_count, rank);
  } else if (uniform_variate < proposals_.cum_.merge) {
    // printf("Before Update_Tempering, Merge\n");
    PerformMerge_Tempering(cur_log_lk_local, cum_rho_map_local, proposed_cum_rho_map_local, change_points_local, log_lk_map_local, proposed_log_lk_map_local, post_rho_map_local, thread_temp, rejection_count, rank);
  } else {
    // fprintf(stderr, "Error in proposal distribution in Update().\n");
  }
    // printf("change points local pointer: %d\n", change_points_local->size());
    // printf("change points pointer: %d\n", change_points_.size());
    // printf("change points local pointer capacity: %d\n", change_points_local->capacity());
    // printf("change points pointer capacity: %d\n", change_points_.capacity());
    // printf("change points local pointer maxsize: %zu\n", change_points_local->max_size());
    // printf("change points pointer maxsize: %zu\n", change_points_.max_size());


}

void Rjmcmc::PerformChange_Tempering(double *cur_log_lk_local, std::vector<double> *cum_rho_map_local, std::vector<double> *proposed_cum_rho_map_local, 
  std::vector<ChangePoint> *change_points_local, LogLkMap *log_lk_map_local, LogLkMap *proposed_log_lk_map_local, PostRhoMap *post_rho_map_local, double thread_temp, std::vector<int> *rejection_count, int rank) {
  // Choose change point uniformly.
  size_t change_point_id = static_cast<size_t>(
      rng_.GetUniformIntGen(0, change_points_local->size() - 2)());
  assert(change_point_id < change_points_local->size() - 1);

  // Compute new rate.
  double old_rate = change_points_local->at(change_point_id).rate_;
  double uniform_variate = rate_change_gen_();
  double new_rate = old_rate * std::exp(uniform_variate);

  // Change state to proposed state.
  change_points_local->at(change_point_id).rate_ = new_rate;

  UpdateCumRhoMap(proposed_cum_rho_map_local, change_point_id);

  size_t left_snp_id = change_points_local->at(change_point_id).snp_id_;
  size_t right_snp_id = change_points_local->at(change_point_id + 1).snp_id_;

  double proposed_log_lk = ProposeLogLk_Tempering(left_snp_id, right_snp_id, log_lk_map_local, proposed_log_lk_map_local);

  // log(lk_new/lk_old * prop_new_to_old/prop_old_to_new
  //   * prior_new/prior_old) (with simplifications)

  double log_mh = thread_temp*acceptance_ratio_.LogMHChange(proposed_log_lk,
                                                *cur_log_lk_local,
                                                new_rate,
                                                old_rate);
  assert(!std::isnan(log_mh));

  double log_accept = std::min(log_mh, 0.0);

  double accept_variate = uniform_gen_();

  if (std::log(accept_variate) < log_accept) {
    // Accept proposed state.
    *cur_log_lk_local = proposed_log_lk;
    
    // printf("Print cur_log_lk_local in perform change %f \n", cur_log_lk_local);

    UpdateLogLkMap_Tempering(left_snp_id, right_snp_id, log_lk_map_local, proposed_log_lk_map_local);
    CopyCumRhoMap(proposed_cum_rho_map_local,
                  cum_rho_map_local,
                  change_points_local->at(change_point_id).snp_id_);

    if (burn_in_p_) {
      accept_log_burn_in_.AcceptChange(true);
    } else {
      accept_log_run_.AcceptChange(true);
    }
  } else {
    // Reject proposed state.
    // Reset state.
    change_points_local->at(change_point_id).rate_ = old_rate;
    CopyCumRhoMap(cum_rho_map_local,
                  proposed_cum_rho_map_local,
                  change_points_local->at(change_point_id).snp_id_);
    if (burn_in_p_) {
      accept_log_burn_in_.AcceptChange(false);
    } else {
      accept_log_run_.AcceptChange(false);
      rejection_count->at(rank) +=1;
    }
  }
}

void Rjmcmc::PerformExtend_Tempering(double *cur_log_lk_local, std::vector<double> *cum_rho_map_local, std::vector<double> *proposed_cum_rho_map_local, 
  std::vector<ChangePoint> *change_points_local, LogLkMap *log_lk_map_local, LogLkMap *proposed_log_lk_map_local, PostRhoMap *post_rho_map_local, double thread_temp, std::vector<int> *rejection_count, int rank) {
  assert(change_points_local->size() >= 2);

  if (change_points_local->size() == 2) {
    return;  // No change points can be removed; return immediately.
  } else {
    // Choose change point uniformly, excluding the first and last
    // change points.
    size_t change_point_id = static_cast<size_t>(
        rng_.GetUniformIntGen(1, change_points_local->size() - 2)());

    assert(change_points_local->at(change_point_id + 1).snp_id_
             - change_points_local->at(change_point_id - 1).snp_id_ > 1);

    // Choose new position uniformly from available interval.
    size_t prev_snp_id = change_points_local->at(change_point_id - 1).snp_id_;
    size_t next_snp_id = change_points_local->at(change_point_id + 1).snp_id_;

    size_t new_snp_id = static_cast<size_t>(
        rng_.GetUniformIntGen(prev_snp_id + 1, next_snp_id - 1)());

    // change state to proposed state
    size_t old_snp_id = change_points_local->at(change_point_id).snp_id_;
    change_points_local->at(change_point_id).snp_id_ = new_snp_id;

    assert(change_point_id > 0);

    UpdateCumRhoMap(
        proposed_cum_rho_map_local,
        old_snp_id <= new_snp_id ? change_point_id - 1 : change_point_id);

    size_t left_snp_id = std::min(old_snp_id, new_snp_id);
    size_t right_snp_id = std::max(old_snp_id, new_snp_id);

    double proposed_log_lk = ProposeLogLk_Tempering(left_snp_id, right_snp_id, log_lk_map_local, proposed_log_lk_map_local);

    double log_mh = thread_temp*acceptance_ratio_.LogMHExtend(proposed_log_lk, *cur_log_lk_local);

    assert(!std::isnan(log_mh));

    double log_accept = std::min(log_mh, 0.0);

    double accept_variate = uniform_gen_();

    if (std::log(accept_variate) < log_accept) {
      // Accept proposed state.
      *cur_log_lk_local = proposed_log_lk;
      UpdateLogLkMap_Tempering(left_snp_id, right_snp_id, log_lk_map_local, proposed_log_lk_map_local);
      CopyCumRhoMap(proposed_cum_rho_map_local,
                    cum_rho_map_local,
                    std::min(old_snp_id, new_snp_id));

      if (burn_in_p_) {
        accept_log_burn_in_.AcceptExtend(true);
      } else {
        accept_log_run_.AcceptExtend(true);
      }
    } else {
      // Reject proposed state.
      // Reset state.
      change_points_local->at(change_point_id).snp_id_ = old_snp_id;
      CopyCumRhoMap(cum_rho_map_local,
                    proposed_cum_rho_map_local,
                    std::min(old_snp_id, new_snp_id));
      if (burn_in_p_) {
        accept_log_burn_in_.AcceptExtend(false);
      } else {
        accept_log_run_.AcceptExtend(false);
        rejection_count->at(rank) +=1;
      }
    }
    return;
  }
}

void Rjmcmc::PerformSplit_Tempering(double *cur_log_lk_local, std::vector<double> *cum_rho_map_local, std::vector<double> *proposed_cum_rho_map_local, 
  std::vector<ChangePoint> *change_points_local, LogLkMap *log_lk_map_local, LogLkMap *proposed_log_lk_map_local, PostRhoMap *post_rho_map_local, double thread_temp, std::vector<int> *rejection_count, int rank) {
  size_t num_snps = snp_pos_.size();
  size_t split_point_snp_id = static_cast<size_t>(
      rng_.GetUniformIntGen(1, num_snps - 2)());

  size_t left_change_point_id, right_change_point_id;

  boost::tie(left_change_point_id, right_change_point_id) =
    BinarySearch(*change_points_local,
                 ChangePoint(split_point_snp_id, 0.0),
                 CompareSnpID());

  assert(right_change_point_id < change_points_local->size());

  if (change_points_local->at(left_change_point_id).snp_id_ == split_point_snp_id) {
    // We chose a snp that's already a change point;
    // return immediately.
    rechosen_snp_ += 1;
    return;
  } else {
    // Choose new rates for splitting up old rate into two new rates.
    double u = uniform_gen_();
    double old_rate = change_points_local->at(left_change_point_id).rate_;

    uint32_t right_pos1 =
      snp_pos_[change_points_local->at(right_change_point_id).snp_id_];
    uint32_t left_pos1 =
      snp_pos_[change_points_local->at(left_change_point_id).snp_id_];
    uint32_t new_pos1 = snp_pos_[split_point_snp_id];

    double new_left_rate =
      old_rate
      * std::pow(u / (1.0 - u),
                 static_cast<double>(right_pos1 - new_pos1)
                   / static_cast<double>(right_pos1 - left_pos1));
    double new_split_rate = ((1.0 - u) / u) * new_left_rate;

    // Needed for acceptance ratio.
    size_t orig_num_change_points = change_points_local->size();
    assert(orig_num_change_points >= 2);

    // Construct proposed state:
    //   Insert new change point.
    //   Insert right block.
    // This invalidates indexes that follow the newly inserted
    // change point (i.e. any index starting from right_change_point_id
    // onward)
    change_points_local->insert(change_points_local->begin() + right_change_point_id,
                          ChangePoint(split_point_snp_id, new_split_rate));
    size_t new_right_change_point_id = right_change_point_id + 1;
    size_t split_change_point_id = right_change_point_id;

    assert(change_points_local->size() >= 2);

    // Change the left block.
    change_points_local->at(left_change_point_id).rate_ = new_left_rate;

    // Log likelihood of proposed state.

    // Compute new log lk.
    UpdateCumRhoMap(proposed_cum_rho_map_local, left_change_point_id);

    size_t left_snp_id = change_points_local->at(left_change_point_id).snp_id_;
    size_t right_snp_id = change_points_local->at(new_right_change_point_id).snp_id_;

    double proposed_log_lk = ProposeLogLk_Tempering(left_snp_id, right_snp_id, log_lk_map_local, proposed_log_lk_map_local);

    double log_mh = thread_temp*acceptance_ratio_.LogMHSplit(proposed_log_lk,
                                     *cur_log_lk_local,
                                     num_snps,
                                     orig_num_change_points,
                                     new_left_rate,
                                     new_split_rate,
                                     old_rate,
                                     proposals_);
    assert(!std::isnan(log_mh));

    double log_accept = std::min(log_mh, 0.0);

    double accept_variate = uniform_gen_();

    if (std::log(accept_variate) < log_accept) {
      // Change state.
      *cur_log_lk_local = proposed_log_lk;
      UpdateLogLkMap_Tempering(left_snp_id, right_snp_id, log_lk_map_local, proposed_log_lk_map_local);
      CopyCumRhoMap(proposed_cum_rho_map_local,
                    cum_rho_map_local,
                    change_points_local->at(left_change_point_id).snp_id_);

      if (burn_in_p_) {
        accept_log_burn_in_.AcceptSplit(true);
      } else {
        accept_log_run_.AcceptSplit(true);
      }
    } else {
      // Don't change state.
      // Reset state.
      change_points_local->erase(change_points_local->begin() + split_change_point_id);
      change_points_local->at(left_change_point_id).rate_ = old_rate;
      CopyCumRhoMap(cum_rho_map_local,
                    proposed_cum_rho_map_local,
                    change_points_local->at(left_change_point_id).snp_id_);

      if (burn_in_p_) {
        accept_log_burn_in_.AcceptSplit(false);
      } else {
        accept_log_run_.AcceptSplit(false);
        rejection_count->at(rank) +=1;
      }
    }

    return;
  }
}

void Rjmcmc::PerformMerge_Tempering(double *cur_log_lk_local, std::vector<double> *cum_rho_map_local, std::vector<double> *proposed_cum_rho_map_local, 
  std::vector<ChangePoint> *change_points_local, LogLkMap *log_lk_map_local, LogLkMap *proposed_log_lk_map_local, PostRhoMap *post_rho_map_local, double thread_temp, std::vector<int> *rejection_count, int rank) {
  assert(change_points_local->size() >= 2);
  if (change_points_local->size() == 2) {
    return;  // No change points can be removed; return immediately.
  } else {
    size_t num_snps = snp_pos_.size();

    // printf("before merge change point: %d and change point local: %d\n", change_points_.size(), change_points_local->size());
    // size_t merge_change_point_id = change_points_local->size() - 2;
    size_t merge_change_point_id = static_cast<size_t>(rng_.GetUniformIntGen(1, change_points_local->size() - 2)());
    // printf("after merge change point, merge_id: %d\n", merge_change_point_id);

    size_t left_change_point_id = merge_change_point_id - 1;
    size_t right_change_point_id = merge_change_point_id + 1;
    assert(right_change_point_id < change_points_local->size());

    // Compute new rate for merged block.
    double left_pos1 =
        snp_pos_[change_points_local->at(left_change_point_id).snp_id_];
    double right_pos1 =
        snp_pos_[change_points_local->at(right_change_point_id).snp_id_];
    double mergePos1 =
        snp_pos_[change_points_local->at(merge_change_point_id).snp_id_];

    double left_rate = change_points_local->at(left_change_point_id).rate_;
    double merge_rate = change_points_local->at(merge_change_point_id).rate_;


    double new_rate =
      std::pow(left_rate,
               static_cast<double>(mergePos1 - left_pos1)
                 / static_cast<double>(right_pos1 - left_pos1))
    * std::pow(merge_rate,
               static_cast<double>(right_pos1 - mergePos1)
                 / static_cast<double>(right_pos1 - left_pos1));

    // Record snp_id of change point to be removed.
    uint32_t merge_snp_id = change_points_local->at(merge_change_point_id).snp_id_;

    size_t orig_num_change_points = change_points_local->size();
    assert(orig_num_change_points >= 2);

    // printf("before erase change points: %d\n", change_points_local->size());
    // Construct proposed state.
    // Remove merge point.
    change_points_local->erase(change_points_local->begin() + merge_change_point_id);

    // printf("after erase change points: %d\n", change_points_local->size());
    // Set rate of left point.
    change_points_local->at(left_change_point_id).rate_ = new_rate;

    // Compute new log lk.
    UpdateCumRhoMap(proposed_cum_rho_map_local, left_change_point_id);

    // printf("Before left_snp_id and right_snp_id: %d, %d", left_change_point_id, right_change_point_id);
    size_t left_snp_id = change_points_local->at(left_change_point_id).snp_id_;
    size_t right_snp_id = (*change_points_local)[right_change_point_id].snp_id_;

    double proposed_log_lk = ProposeLogLk_Tempering(left_snp_id, right_snp_id, log_lk_map_local, proposed_log_lk_map_local);

    // Log Metropolis-Hasting ratio:
    double log_mh = thread_temp*acceptance_ratio_.LogMHMerge(proposed_log_lk,
                                   *cur_log_lk_local,
                                   num_snps,
                                   orig_num_change_points,
                                   left_rate,
                                   merge_rate,
                                   new_rate,
                                   proposals_);
    assert(!std::isnan(log_mh));

    double log_accept = std::min(log_mh, 0.0);

    double accept_variate = uniform_gen_();

    if (std::log(accept_variate) < log_accept) {
      // Change state.
      *cur_log_lk_local = proposed_log_lk;
      UpdateLogLkMap_Tempering(left_snp_id, right_snp_id, log_lk_map_local, proposed_log_lk_map_local);
      CopyCumRhoMap(proposed_cum_rho_map_local,
                    cum_rho_map_local,
                    change_points_local->at(left_change_point_id).snp_id_);

      if (burn_in_p_) {
        accept_log_burn_in_.AcceptMerge(true);
      } else {
        accept_log_run_.AcceptMerge(true);
      }
    } else {
      // Reject proposed state.
      // Reset state.
      change_points_local->insert(
          change_points_local->begin() + merge_change_point_id,
          ChangePoint(merge_snp_id, merge_rate));

      change_points_local->at(left_change_point_id).rate_ = left_rate;

      CopyCumRhoMap(cum_rho_map_local,
                    proposed_cum_rho_map_local,
                    change_points_local->at(left_change_point_id).snp_id_);

      // Reset log lk.
      if (burn_in_p_) {
        accept_log_burn_in_.AcceptMerge(false);
      } else {
        accept_log_run_.AcceptMerge(false);
        rejection_count->at(rank) +=1;
      }
    }

    return;
  }
}

double Rjmcmc::ProposeLogLk_Tempering(size_t left_snp_id, size_t right_snp_id, LogLkMap *log_lk_map_local, LogLkMap *proposed_log_lk_map_local) {
  size_t begin_snp_id =
    left_snp_id >= window_size_-1 ? left_snp_id - (window_size_ - 1) : 0;
  size_t end_snp_id = right_snp_id;

  double proposed_log_lk = cur_log_lk;

  for (size_t site0 = begin_snp_id; site0 < end_snp_id; ++site0) {
    size_t max_site1 = std::min(site0 + window_size_, snp_pos_.size());
    for (size_t site1 = site0 + 1; site1 < max_site1; ++site1) {
      // Subtract out old log lk.
      proposed_log_lk -= log_lk_map_[site0 * window_size_ + (site1 - site0)];

      // Compute new log lk.
      double genetic_distance =
        proposed_cum_rho_map_[site1] - proposed_cum_rho_map_[site0];
      double conf_log_lk =
        log_lk_computer_.ComputeLogLkFromSites(genetic_distance,
                                               site0,
                                               site1);

      // Update log lk for (site0, site1).
      proposed_log_lk_map_[site0 * window_size_ + (site1 - site0)] = conf_log_lk;

      // Add new log lk.
      proposed_log_lk += conf_log_lk;
    }
  }

  return proposed_log_lk;
}

void Rjmcmc::UpdateLogLkMap_Tempering(size_t left_snp_id, size_t right_snp_id, LogLkMap *log_lk_map_local, LogLkMap *proposed_log_lk_map_local) {
  size_t begin_snp_id =
      left_snp_id >= window_size_ - 1 ? left_snp_id - (window_size_ - 1) : 0;
  size_t end_snp_id = right_snp_id;
  assert(end_snp_id <= snp_pos_.size());

  for (size_t site0 = begin_snp_id; site0 < end_snp_id; ++site0) {
    size_t max_site1 = std::min(site0 + window_size_, snp_pos_.size());
    for (size_t site1 = site0 + 1; site1 < max_site1; ++site1) {
      log_lk_map_[site0 * window_size_ + (site1 - site0)] = proposed_log_lk_map_[site0 * window_size_ + (site1 - site0)];
    }
  }
}

void Rjmcmc::RecordSample_Tempering(std::vector<ChangePoint> *change_points_local) {
  assert(change_points_local->size() >= 2);

  assert(partition_end_ > partition_start_);
  assert(partition_end_ >= 2);
  assert(partition_end_ - partition_start_ >= 2);

  assert(snp_pos_.size() >= 2);
  assert(snp_pos_.size() >= partition_end_);

  sample_store_.push_back(std::vector<std::pair<uint64_t, double> >());

  // Special case: first change point in partition.
  //   Find rightmost change point to the left of or equal
  //   to partition_start_ to get rate of partition_start_.
  size_t first_index = 0;
  while (!(change_points_local->at(first_index).snp_id_ <= partition_start_ &&
           change_points_local->at(first_index + 1).snp_id_ > partition_start_)) {
    assert(first_index + 1 < change_points_local->size());
    ++first_index;
  }
  sample_store_.back().push_back(
      std::make_pair(snp_pos_[partition_start_],
                     change_points_local->at(first_index).rate_));

  // Special case: last change point in partition.
  //   Find rightmost change point strictly to the left of
  //   partition_end_ - 1.
  size_t last_index = 0;
  while (!(change_points_local->at(last_index).snp_id_ < partition_end_ - 1 &&
           change_points_local->at(last_index + 1).snp_id_ >= partition_end_ - 1)) {
    assert(last_index + 1 < change_points_local->size());
    ++last_index;
  }

  assert(first_index <= last_index);

  // Record change points up to and including last_index, starting
  // from the change point right after first_index.
  for (size_t i = first_index + 1; i < last_index + 1; ++i) {
    sample_store_.back().push_back(
        std::make_pair(snp_pos_[change_points_local->at(i).snp_id_],
                       change_points_local->at(i).rate_));
  }

  sample_store_.back().push_back(
      std::make_pair(snp_pos_[partition_end_-1], -1.0));
}


/************************************************************Parallel Tempering Methods END**********************************/

void Rjmcmc::Update() {
  double uniform_variate = uniform_gen_();
  if (uniform_variate < proposals_.cum_.change) {
    PerformChange();
  } else if (uniform_variate < proposals_.cum_.extend) {
    PerformExtend();
  } else if (uniform_variate < proposals_.cum_.split) {
    PerformSplit();
  } else if (uniform_variate < proposals_.cum_.merge) {
    PerformMerge();
  } else {
    // fprintf(stderr, "Error in proposal distribution in Update().\n");
  }

}

void Rjmcmc::PerformChange() {
  // Choose change point uniformly.
  size_t change_point_id = static_cast<size_t>(
      rng_.GetUniformIntGen(0, change_points_.size() - 2)());
  assert(change_point_id < change_points_.size() - 1);

  // Compute new rate.
  double old_rate = change_points_[change_point_id].rate_;
  double uniform_variate = rate_change_gen_();
  double new_rate = old_rate * std::exp(uniform_variate);

  // Change state to proposed state.
  change_points_[change_point_id].rate_ = new_rate;

  UpdateCumRhoMap(&proposed_cum_rho_map_, change_point_id);

  size_t left_snp_id = change_points_[change_point_id].snp_id_;
  size_t right_snp_id = change_points_[change_point_id + 1].snp_id_;

  double proposed_log_lk = ProposeLogLk(left_snp_id, right_snp_id);

  // log(lk_new/lk_old * prop_new_to_old/prop_old_to_new
  //   * prior_new/prior_old) (with simplifications)

  double log_mh = acceptance_ratio_.LogMHChange(proposed_log_lk,
                                                cur_log_lk,
                                                new_rate,
                                                old_rate);
  assert(!std::isnan(log_mh));

  double log_accept = std::min(log_mh, 0.0);

  double accept_variate = uniform_gen_();

  if (std::log(accept_variate) < log_accept) {
    // Accept proposed state.
    cur_log_lk = proposed_log_lk;
    UpdateLogLkMap(left_snp_id, right_snp_id);
    CopyCumRhoMap(&proposed_cum_rho_map_,
                  &cum_rho_map_,
                  change_points_[change_point_id].snp_id_);

    if (burn_in_p_) {
      accept_log_burn_in_.AcceptChange(true);
    } else {
      accept_log_run_.AcceptChange(true);
    }
  } else {
    // Reject proposed state.
    // Reset state.
    change_points_[change_point_id].rate_ = old_rate;
    CopyCumRhoMap(&cum_rho_map_,
                  &proposed_cum_rho_map_,
                  change_points_[change_point_id].snp_id_);
    if (burn_in_p_) {
      accept_log_burn_in_.AcceptChange(false);
    } else {
      accept_log_run_.AcceptChange(false);
    }
  }
}

void Rjmcmc::PerformExtend() {
  assert(change_points_.size() >= 2);

  if (change_points_.size() == 2) {
    return;  // No change points can be removed; return immediately.
  } else {
    // Choose change point uniformly, excluding the first and last
    // change points.
    size_t change_point_id = static_cast<size_t>(
        rng_.GetUniformIntGen(1, change_points_.size() - 2)());

    assert(change_points_[change_point_id + 1].snp_id_
             - change_points_[change_point_id - 1].snp_id_ > 1);

    // Choose new position uniformly from available interval.
    size_t prev_snp_id = change_points_[change_point_id - 1].snp_id_;
    size_t next_snp_id = change_points_[change_point_id + 1].snp_id_;

    size_t new_snp_id = static_cast<size_t>(
        rng_.GetUniformIntGen(prev_snp_id + 1, next_snp_id - 1)());

    // change state to proposed state
    size_t old_snp_id = change_points_[change_point_id].snp_id_;
    change_points_[change_point_id].snp_id_ = new_snp_id;

    assert(change_point_id > 0);

    UpdateCumRhoMap(
        &proposed_cum_rho_map_,
        old_snp_id <= new_snp_id ? change_point_id - 1 : change_point_id);

    size_t left_snp_id = std::min(old_snp_id, new_snp_id);
    size_t right_snp_id = std::max(old_snp_id, new_snp_id);

    double proposed_log_lk = ProposeLogLk(left_snp_id, right_snp_id);

    double log_mh =
      acceptance_ratio_.LogMHExtend(proposed_log_lk, cur_log_lk);

    assert(!std::isnan(log_mh));

    double log_accept = std::min(log_mh, 0.0);

    double accept_variate = uniform_gen_();

    if (std::log(accept_variate) < log_accept) {
      // Accept proposed state.
      cur_log_lk = proposed_log_lk;
      UpdateLogLkMap(left_snp_id, right_snp_id);
      CopyCumRhoMap(&proposed_cum_rho_map_,
                    &cum_rho_map_,
                    std::min(old_snp_id, new_snp_id));

      if (burn_in_p_) {
        accept_log_burn_in_.AcceptExtend(true);
      } else {
        accept_log_run_.AcceptExtend(true);
      }
    } else {
      // Reject proposed state.
      // Reset state.
      change_points_[change_point_id].snp_id_ = old_snp_id;
      CopyCumRhoMap(&cum_rho_map_,
                    &proposed_cum_rho_map_,
                    std::min(old_snp_id, new_snp_id));
      if (burn_in_p_) {
        accept_log_burn_in_.AcceptExtend(false);
      } else {
        accept_log_run_.AcceptExtend(false);
      }
    }
    return;
  }
}

void Rjmcmc::PerformSplit() {
  size_t num_snps = snp_pos_.size();
  size_t split_point_snp_id = static_cast<size_t>(
      rng_.GetUniformIntGen(1, num_snps - 2)());

  size_t left_change_point_id, right_change_point_id;

  boost::tie(left_change_point_id, right_change_point_id) =
    BinarySearch(change_points_,
                 ChangePoint(split_point_snp_id, 0.0),
                 CompareSnpID());

  assert(right_change_point_id < change_points_.size());

  if (change_points_[left_change_point_id].snp_id_ == split_point_snp_id) {
    // We chose a snp that's already a change point;
    // return immediately.
    rechosen_snp_ += 1;
    return;
  } else {
    // Choose new rates for splitting up old rate into two new rates.
    double u = uniform_gen_();
    double old_rate = change_points_[left_change_point_id].rate_;

    uint32_t right_pos1 =
      snp_pos_[change_points_[right_change_point_id].snp_id_];
    uint32_t left_pos1 =
      snp_pos_[change_points_[left_change_point_id].snp_id_];
    uint32_t new_pos1 = snp_pos_[split_point_snp_id];

    double new_left_rate =
      old_rate
      * std::pow(u / (1.0 - u),
                 static_cast<double>(right_pos1 - new_pos1)
                   / static_cast<double>(right_pos1 - left_pos1));
    double new_split_rate = ((1.0 - u) / u) * new_left_rate;

    // Needed for acceptance ratio.
    size_t orig_num_change_points = change_points_.size();
    assert(orig_num_change_points >= 2);

    // Construct proposed state:
    //   Insert new change point.
    //   Insert right block.
    // This invalidates indexes that follow the newly inserted
    // change point (i.e. any index starting from right_change_point_id
    // onward)
    change_points_.insert(change_points_.begin() + right_change_point_id,
                          ChangePoint(split_point_snp_id, new_split_rate));
    size_t new_right_change_point_id = right_change_point_id + 1;
    size_t split_change_point_id = right_change_point_id;

    assert(change_points_.size() >= 2);

    // Change the left block.
    change_points_[left_change_point_id].rate_ = new_left_rate;

    // Log likelihood of proposed state.

    // Compute new log lk.
    UpdateCumRhoMap(&proposed_cum_rho_map_, left_change_point_id);

    size_t left_snp_id = change_points_[left_change_point_id].snp_id_;
    size_t right_snp_id = change_points_[new_right_change_point_id].snp_id_;

    double proposed_log_lk = ProposeLogLk(left_snp_id, right_snp_id);

    double log_mh =
        acceptance_ratio_.LogMHSplit(proposed_log_lk,
                                     cur_log_lk,
                                     num_snps,
                                     orig_num_change_points,
                                     new_left_rate,
                                     new_split_rate,
                                     old_rate,
                                     proposals_);
    assert(!std::isnan(log_mh));

    double log_accept = std::min(log_mh, 0.0);

    double accept_variate = uniform_gen_();

    if (std::log(accept_variate) < log_accept) {
      // Change state.
      cur_log_lk = proposed_log_lk;
      UpdateLogLkMap(left_snp_id, right_snp_id);
      CopyCumRhoMap(&proposed_cum_rho_map_,
                    &cum_rho_map_,
                    change_points_[left_change_point_id].snp_id_);

      if (burn_in_p_) {
        accept_log_burn_in_.AcceptSplit(true);
      } else {
        accept_log_run_.AcceptSplit(true);
      }
    } else {
      // Don't change state.
      // Reset state.
      change_points_.erase(change_points_.begin() + split_change_point_id);
      change_points_[left_change_point_id].rate_ = old_rate;
      CopyCumRhoMap(&cum_rho_map_,
                    &proposed_cum_rho_map_,
                    change_points_[left_change_point_id].snp_id_);

      if (burn_in_p_) {
        accept_log_burn_in_.AcceptSplit(false);
      } else {
        accept_log_run_.AcceptSplit(false);
      }
    }

    return;
  }
}

void Rjmcmc::PerformMerge() {
  assert(change_points_.size() >= 2);
  if (change_points_.size() == 2) {
    return;  // No change points can be removed; return immediately.
  } else {
    size_t num_snps = snp_pos_.size();
    size_t merge_change_point_id = static_cast<size_t>(
        rng_.GetUniformIntGen(1, change_points_.size() - 2)());

    size_t left_change_point_id = merge_change_point_id - 1;
    size_t right_change_point_id = merge_change_point_id + 1;
    assert(right_change_point_id < change_points_.size());

    // Compute new rate for merged block.
    double left_pos1 =
        snp_pos_[change_points_[left_change_point_id].snp_id_];
    double right_pos1 =
        snp_pos_[change_points_[right_change_point_id].snp_id_];
    double mergePos1 =
        snp_pos_[change_points_[merge_change_point_id].snp_id_];

    double left_rate = change_points_[left_change_point_id].rate_;
    double merge_rate = change_points_[merge_change_point_id].rate_;

    double new_rate =
      std::pow(left_rate,
               static_cast<double>(mergePos1 - left_pos1)
                 / static_cast<double>(right_pos1 - left_pos1))
    * std::pow(merge_rate,
               static_cast<double>(right_pos1 - mergePos1)
                 / static_cast<double>(right_pos1 - left_pos1));

    // Record snp_id of change point to be removed.
    uint32_t merge_snp_id = change_points_[merge_change_point_id].snp_id_;

    size_t orig_num_change_points = change_points_.size();
    assert(orig_num_change_points >= 2);

    // Construct proposed state.
    // Remove merge point.
    change_points_.erase(change_points_.begin() + merge_change_point_id);

    // Set rate of left point.
    change_points_[left_change_point_id].rate_ = new_rate;

    // Compute new log lk.
    UpdateCumRhoMap(&proposed_cum_rho_map_, left_change_point_id);

    size_t left_snp_id = change_points_[left_change_point_id].snp_id_;
    size_t right_snp_id = change_points_[right_change_point_id].snp_id_;

    double proposed_log_lk = ProposeLogLk(left_snp_id, right_snp_id);

    // Log Metropolis-Hasting ratio:
    double log_mh =
      acceptance_ratio_.LogMHMerge(proposed_log_lk,
                                   cur_log_lk,
                                   num_snps,
                                   orig_num_change_points,
                                   left_rate,
                                   merge_rate,
                                   new_rate,
                                   proposals_);
    assert(!std::isnan(log_mh));

    double log_accept = std::min(log_mh, 0.0);

    double accept_variate = uniform_gen_();

    if (std::log(accept_variate) < log_accept) {
      // Change state.
      cur_log_lk = proposed_log_lk;
      UpdateLogLkMap(left_snp_id, right_snp_id);
      CopyCumRhoMap(&proposed_cum_rho_map_,
                    &cum_rho_map_,
                    change_points_[left_change_point_id].snp_id_);

      if (burn_in_p_) {
        accept_log_burn_in_.AcceptMerge(true);
      } else {
        accept_log_run_.AcceptMerge(true);
      }
    } else {
      // Reject proposed state.
      // Reset state.
      change_points_.insert(
          change_points_.begin() + merge_change_point_id,
          ChangePoint(merge_snp_id, merge_rate));

      change_points_[left_change_point_id].rate_ = left_rate;

      CopyCumRhoMap(&cum_rho_map_,
                    &proposed_cum_rho_map_,
                    change_points_[left_change_point_id].snp_id_);

      // Reset log lk.
      if (burn_in_p_) {
        accept_log_burn_in_.AcceptMerge(false);
      } else {
        accept_log_run_.AcceptMerge(false);
      }
    }

    return;
  }
}

double Rjmcmc::ProposeLogLk(size_t left_snp_id, size_t right_snp_id) {
  size_t begin_snp_id =
    left_snp_id >= window_size_-1 ? left_snp_id - (window_size_ - 1) : 0;
  size_t end_snp_id = right_snp_id;

  double proposed_log_lk = cur_log_lk;

  for (size_t site0 = begin_snp_id; site0 < end_snp_id; ++site0) {
    size_t max_site1 = std::min(site0 + window_size_, snp_pos_.size());
    for (size_t site1 = site0 + 1; site1 < max_site1; ++site1) {
      // Subtract out old log lk.
      proposed_log_lk -= log_lk_map_[site0 * window_size_ + (site1 - site0)];

      // Compute new log lk.
      double genetic_distance =
        proposed_cum_rho_map_[site1] - proposed_cum_rho_map_[site0];
      double conf_log_lk =
        log_lk_computer_.ComputeLogLkFromSites(genetic_distance,
                                               site0,
                                               site1);

      // Update log lk for (site0, site1).
      proposed_log_lk_map_[site0 * window_size_ + (site1 - site0)] =
        conf_log_lk;

      // Add new log lk.
      proposed_log_lk += conf_log_lk;
    }
  }

  return proposed_log_lk;
}

void Rjmcmc::UpdateLogLkMap(size_t left_snp_id, size_t right_snp_id) {
  size_t begin_snp_id =
      left_snp_id >= window_size_ - 1 ? left_snp_id - (window_size_ - 1) : 0;
  size_t end_snp_id = right_snp_id;
  assert(end_snp_id <= snp_pos_.size());

  for (size_t site0 = begin_snp_id; site0 < end_snp_id; ++site0) {
    size_t max_site1 = std::min(site0 + window_size_, snp_pos_.size());
    for (size_t site1 = site0 + 1; site1 < max_site1; ++site1) {
      log_lk_map_[site0 * window_size_ + (site1 - site0)] =
        proposed_log_lk_map_[site0 * window_size_ + (site1 - site0)];
    }
  }
}

void Rjmcmc::UpdateCumRhoMap(std::vector<double> *in_cum_rho_map,
                             size_t begin_change_point_id) {
  double rho_sum =
    (*in_cum_rho_map)[change_points_[begin_change_point_id].snp_id_];

  for (size_t change_point_id = begin_change_point_id;
       change_point_id < change_points_.size() - 1;
       ++change_point_id) {
    for (size_t snp_id = change_points_[change_point_id].snp_id_;
         snp_id < change_points_[change_point_id + 1].snp_id_;
         ++snp_id) {
       assert(snp_id < snp_pos_.size());
       (*in_cum_rho_map)[snp_id] = rho_sum;
       rho_sum += static_cast<double>(snp_pos_[snp_id + 1] - snp_pos_[snp_id])
                * change_points_[change_point_id].rate_;
       assert(change_points_[change_point_id].rate_ >= 0.0);
    }
  }
  in_cum_rho_map->back() = rho_sum;
}

void Rjmcmc::CopyCumRhoMap(std::vector<double> *source,
                           std::vector<double> *dest,
                           size_t start_snp_id) const {
  for (size_t snp_id = start_snp_id; snp_id < snp_pos_.size(); ++snp_id) {
    (*dest)[snp_id] = (*source)[snp_id];
  }
}

void Rjmcmc::RecordSample() {
  assert(change_points_.size() >= 2);

  assert(partition_end_ > partition_start_);
  assert(partition_end_ >= 2);
  assert(partition_end_ - partition_start_ >= 2);

  assert(snp_pos_.size() >= 2);
  assert(snp_pos_.size() >= partition_end_);

  sample_store_.push_back(std::vector<std::pair<uint64_t, double> >());

  // Special case: first change point in partition.
  //   Find rightmost change point to the left of or equal
  //   to partition_start_ to get rate of partition_start_.
  size_t first_index = 0;
  while (!(change_points_[first_index].snp_id_ <= partition_start_ &&
           change_points_[first_index + 1].snp_id_ > partition_start_)) {
    assert(first_index + 1 < change_points_.size());
    ++first_index;
  }
  sample_store_.back().push_back(
      std::make_pair(snp_pos_[partition_start_],
                     change_points_[first_index].rate_));

  // Special case: last change point in partition.
  //   Find rightmost change point strictly to the left of
  //   partition_end_ - 1.
  size_t last_index = 0;
  while (!(change_points_[last_index].snp_id_ < partition_end_ - 1 &&
           change_points_[last_index + 1].snp_id_ >= partition_end_ - 1)) {
    assert(last_index + 1 < change_points_.size());
    ++last_index;
  }

  assert(first_index <= last_index);

  // Record change points up to and including last_index, starting
  // from the change point right after first_index.
  for (size_t i = first_index + 1; i < last_index + 1; ++i) {
    sample_store_.back().push_back(
        std::make_pair(snp_pos_[change_points_[i].snp_id_],
                       change_points_[i].rate_));
  }

  sample_store_.back().push_back(
      std::make_pair(snp_pos_[partition_end_-1], -1.0));
}
