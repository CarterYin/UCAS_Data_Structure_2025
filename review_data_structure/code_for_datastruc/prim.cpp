// 使用示例：从顶点0出发，启动Prim算法
// g->pfs(0, PrimPU<char, Rank>());

// Prim算法的顶点优先级更新器
template <typename Tv, typename Te> 
struct PrimPU {
    // 对v的每个尚未被发现的邻居u，按Prim策略做松弛操作
    virtual void operator()(Graph<Tv, Te>* g, Rank v, Rank u) {
        if (UNDISCOVERED != g->status(u)) return;  // 尚未被发现的邻居u，按
        if (g->priority(u) > g->weight(v, u)) {  // Prim
            g->priority(u) = g->weight(v, u);  // 策略
            g->parent(u) = v;  // 做松弛
        }
    }
};