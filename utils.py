''' Utility functions for learning scripts '''

def log_scalars(writer, agent, step):
    
    # Agent (PPO) 
    if hasattr(agent, "approx_kl"):         writer.add_scalar("algo/KL", agent.approx_kl, step)
    if hasattr(agent, "p_loss"):            writer.add_scalar("algo/policy loss", agent.p_loss, step)
    if hasattr(agent, "v_loss"):            writer.add_scalar("algo/value loss", agent.v_loss, step)
    if hasattr(agent, "entropy_loss"):      writer.add_scalar("algo/entropy loss", agent.entropy_loss, step)
    if hasattr(agent, "loss"):              writer.add_scalar("algo/total loss", agent.loss, step)
    if hasattr(agent, "clipfracs"):         writer.add_scalar("algo/clipfrac", agent.clipfracs, step)
    if hasattr(agent, "explained_var"):     writer.add_scalar("algo/explained var", agent.explained_var, step)
    if hasattr(agent, "ppo_updates"):       writer.add_scalar("algo/ppo updates", agent.ppo_updates, step)
    if hasattr(agent, "grad_norm"):         writer.add_scalar("algo/gradient norm", agent.grad_norm, step)
    if hasattr(agent, "actor_grad_norm"):   writer.add_scalar("algo/actor_gradient norm", agent.actor_grad_norm, step)
    if hasattr(agent, "critic_grad_norm"):  writer.add_scalar("algo/critic_gradient norm", agent.critic_grad_norm, step)
    if hasattr(agent, "adv_rtn_corr"):      writer.add_scalar("algo/critic adv rtn correlation", agent.adv_rtn_corr, step)
    if hasattr(agent, "actor"):
        if hasattr(agent.actor, "logstd"):  writer.add_scalar("algo/actor action std", agent.actor.logstd.exp().mean(), step)

    # World Model / Other Model
    if hasattr(agent, "world_loss"):        writer.add_scalar("world/model loss", agent.world_loss, step)
    if hasattr(agent, "world_epochs_cnt"):  writer.add_scalar("world/model epoch count", agent.world_epochs_cnt, step)
    if hasattr(agent, "world_kl_loss"):     writer.add_scalar("world/model kl div loss", agent.world_kl_loss, step)
    if hasattr(agent, "world_nan_loss_cnt"):writer.add_scalar("world/nan losses", agent.world_nan_loss_cnt, step)
    if hasattr(agent, "world_mbtrain_pct"): writer.add_scalar("world/mbatch % trained", agent.world_mbtrain_pct, step)
    if hasattr(agent, "idm_loss"):          writer.add_scalar("world/inverse dynamics loss", agent.idm_loss, step)
    if hasattr(agent, "idm_epochs_cnt"):    writer.add_scalar("world/inverse dynamics epochs", agent.idm_epochs_cnt, step)
    if hasattr(agent, "spvd_kl_div"):       writer.add_scalar("world/supervised kl div", agent.spvd_kl_div, step)
    if hasattr(agent, "spvd_epochs_cnt"):   writer.add_scalar("world/supervised epochs", agent.spvd_epochs_cnt, step)
    if hasattr(agent, "wm_loss"):           writer.add_scalar("world/supervised loss", agent.wm_loss, step)
    if hasattr(agent, "obs_loss"):          writer.add_scalar("world/supervised obs loss", agent.obs_loss, step)
    if hasattr(agent, "r_loss"):            writer.add_scalar("world/supervised r loss", agent.r_loss, step)
    if hasattr(agent, "d_loss"):            writer.add_scalar("world/supervised d loss", agent.d_loss, step)
    if hasattr(agent, "wm_grad_norm"):      writer.add_scalar("world/gradient norm", agent.wm_grad_norm, step)
    if hasattr(agent, "env_mean_ed"):       writer.add_scalar("world/mean env euclidean distance", agent.env_mean_ed, step)
    if hasattr(agent, "pred_mean_ed"):      writer.add_scalar("world/mean pred euclidean distance", agent.pred_mean_ed, step)

    # Miscellaneous metrics
    if hasattr(agent, "actor_avg_wgt_mag"): writer.add_scalar("plasticity/actor avg weight magnitude", agent.actor_avg_wgt_mag, step)
    if hasattr(agent, "critic_avg_wgt_mag"):writer.add_scalar("plasticity/critic avg weight magnitude", agent.critic_avg_wgt_mag, step)
    if hasattr(agent, "wm_avg_wgt_mag"):    writer.add_scalar("plasticity/wm avg weight magnitude", agent.wm_avg_wgt_mag, step)
    if hasattr(agent, "actor_dead_pct"):    writer.add_scalar("plasticity/actor dead units %", agent.actor_dead_pct, step)
    if hasattr(agent, "critic_dead_pct"):   writer.add_scalar("plasticity/critic dead units %", agent.critic_dead_pct, step)
    if hasattr(agent, "wm_dead_pct"):       writer.add_scalar("plasticity/wm dead units %", agent.wm_dead_pct, step)