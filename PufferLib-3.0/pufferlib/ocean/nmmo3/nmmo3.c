#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "puffernet.h"
#include "nmmo3.h"
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <signal.h>
#if defined(PLATFORM_WEB)
#include <emscripten/emscripten.h>
#include <emscripten/websocket.h>
#endif

// Only rens a few agents in the C
// version, and reduces for web.
// You can run the full 1024 on GPU
// with PyTorch.
#if defined(PLATFORM_WEB)
    #define NUM_AGENTS 4
#else
    #define NUM_AGENTS 16
#endif


#define NMMO_SERVER_PORT   7777
#define NMMO_MAX_CLIENTS   16

// 'NMM3' as big-endian ASCII
#define NMMO_MAGIC 0x4E4D4D33u

#pragma pack(push, 1)
typedef struct {
    uint32_t magic; // NMMO_MAGIC
    uint16_t version; // 1
    uint16_t reserved; // 0
    int32_t width;
    int32_t height;
    int32_t num_players;
    int32_t num_enemies;
} NetMapHeaderV1;

typedef struct {
    uint32_t tick;
    uint32_t n_entities; // number of NetEntityV1 records following
} NetTickHeaderV1;

typedef struct {
    int32_t id; // stable pid: [0..num_players+num_enemies)
    int32_t type; // ENTITY_PLAYER / ENTITY_ENEMY
    int32_t r, c;
    int32_t hp, hp_max;
    int32_t comb_lvl;
    int32_t element;
    int32_t anim;
    int32_t dir;
} NetEntityV1;
#pragma pack(pop)

typedef struct {
    int listen_fd;
    int client_fds[NMMO_MAX_CLIENTS];
    int num_clients;
} NMMOServer;


#if !defined(PLATFORM_WEB)
static int nmmo_set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags == -1) return -1;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

// Send all bytes, but if the socket would block, we skip (drop this update).
// On hard error, return -1 so caller can drop the client.
static int nmmo_send_all_nonblocking(int fd, const void *buf, size_t len) {
    const uint8_t *p = (const uint8_t *) buf;
    while (len > 0) {
        int flags = 0;
#ifdef MSG_NOSIGNAL
        flags |= MSG_NOSIGNAL;   // Linux: suppress SIGPIPE per-call
#endif
        ssize_t n = send(fd, p, len, flags);
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                return 0; // skip remainder this tick
            }
            // EPIPE, ECONNRESET, etc. => drop client
            return -1;
        }
        if (n == 0) return -1;
        p   += (size_t) n;
        len -= (size_t) n;
    }
    return 0;
}

static int nmmo_server_init(NMMOServer *srv, uint16_t port) {
    memset(srv, 0, sizeof(*srv));
    srv->listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (srv->listen_fd < 0) {
        perror("socket");
        return -1;
    }

    int yes = 1;
   if (setsockopt(srv->listen_fd, SOL_SOCKET, SO_REUSEADDR,
                   &yes, sizeof(yes)) < 0) {
        perror("setsockopt(SO_REUSEADDR)");
        // not fatal, keep going if you want
    }

    #ifdef SO_NOSIGPIPE
    // On macOS/BSD: prevent SIGPIPE on this listening socket
    if (setsockopt(srv->listen_fd, SOL_SOCKET, SO_NOSIGPIPE,
                   &yes, sizeof(yes)) < 0) {
        perror("setsockopt(SO_NOSIGPIPE listen_fd)");
        // not fatal
    }
    #endif

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);

    if (bind(srv->listen_fd, (struct sockaddr *) &addr, sizeof(addr)) < 0) {
        perror("bind");
        close(srv->listen_fd);
        srv->listen_fd = -1;
        return -1;
    }
    if (listen(srv->listen_fd, NMMO_MAX_CLIENTS) < 0) {
        perror("listen");
        close(srv->listen_fd);
        srv->listen_fd = -1;
        return -1;
    }
    if (nmmo_set_nonblocking(srv->listen_fd) < 0) {
        perror("fcntl(listen_fd)");
        close(srv->listen_fd);
        srv->listen_fd = -1;
        return -1;
    }

    printf("[nmmo-server] listening on 0.0.0.0:%u\n", (unsigned) port);
    return 0;
}

static size_t nmmo_serialize_tick_v1(const MMO *env, uint8_t *buf, size_t cap) {
    uint32_t n_entities = (uint32_t) (env->num_players + env->num_enemies);
    size_t needed = sizeof(NetTickHeaderV1) + (size_t) n_entities * sizeof(NetEntityV1);
    if (needed > cap) return 0;

    NetTickHeaderV1 *hdr = (NetTickHeaderV1 *) buf;
    hdr->tick = htonl((uint32_t) env->tick);
    hdr->n_entities = htonl(n_entities);

    NetEntityV1 *out = (NetEntityV1 *) (buf + sizeof(NetTickHeaderV1));

    // Players: pid == index
    for (int i = 0; i < env->num_players; i++) {
        const Entity *e = &env->players[i];
        out[i].id = htonl((int32_t) i);
        out[i].type = htonl((int32_t) e->type);
        out[i].r = htonl((int32_t) e->r);
        out[i].c = htonl((int32_t) e->c);
        out[i].hp = htonl((int32_t) e->hp);
        out[i].hp_max = htonl((int32_t) e->hp_max);
        out[i].comb_lvl = htonl((int32_t) e->comb_lvl);
        out[i].element = htonl((int32_t) e->element);
        out[i].anim = htonl((int32_t) e->anim);
        out[i].dir = htonl((int32_t) e->dir);
    }

    // Enemies: pid == num_players + index
    for (int i = 0; i < env->num_enemies; i++) {
        const Entity *e = &env->enemies[i];
        int idx = env->num_players + i;
        out[idx].id = htonl((int32_t) idx);
        out[idx].type = htonl((int32_t) e->type);
        out[idx].r = htonl((int32_t) e->r);
        out[idx].c = htonl((int32_t) e->c);
        out[idx].hp = htonl((int32_t) e->hp);
        out[idx].hp_max = htonl((int32_t) e->hp_max);
        out[idx].comb_lvl = htonl((int32_t) e->comb_lvl);
        out[idx].element = htonl((int32_t) e->element);
        out[idx].anim = htonl((int32_t) e->anim);
        out[idx].dir = htonl((int32_t) e->dir);
    }

    return needed;
}

static int nmmo_send_map_v1(int fd, const MMO *env) {
    NetMapHeaderV1 mh;
    memset(&mh, 0, sizeof(mh));
    mh.magic = htonl(NMMO_MAGIC);
    mh.version = htons(1);
    mh.reserved = htons(0);
    mh.width = htonl((int32_t) env->width);
    mh.height = htonl((int32_t) env->height);
    mh.num_players = htonl((int32_t) env->num_players);
    mh.num_enemies = htonl((int32_t) env->num_enemies);

    // Header + terrain
    if (nmmo_send_all_nonblocking(fd, &mh, sizeof(mh)) < 0) return -1;
    size_t terrain_bytes = (size_t) env->width * (size_t) env->height;
    if (nmmo_send_all_nonblocking(fd, env->terrain, terrain_bytes) < 0) return -1;
    return 0;
}

static int nmmo_send_tick_v1(int fd, const MMO *env) {
    uint8_t buf[65536];
    size_t len = nmmo_serialize_tick_v1(env, buf, sizeof(buf));
    if (len == 0) return -1;
    return nmmo_send_all_nonblocking(fd, buf, len);
}

// Accept any pending clients; on connect, send map + an immediate tick snapshot
static void nmmo_server_accept_new(NMMOServer *srv, const MMO *env) {
    if (srv->listen_fd < 0) return;

    for (;;) {
        struct sockaddr_in cli_addr;
        socklen_t cli_len = sizeof(cli_addr);
        int cfd = accept(srv->listen_fd, (struct sockaddr *) &cli_addr, &cli_len);
        if (cfd < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) break;
            perror("accept");
            break;
        }

        if (nmmo_set_nonblocking(cfd) < 0) {
            perror("fcntl(client_fd)");
            close(cfd);
            continue;
        }

#ifdef SO_NOSIGPIPE
        {
            int yes = 1;
            if (setsockopt(cfd, SOL_SOCKET, SO_NOSIGPIPE,
                           &yes, sizeof(yes)) < 0) {
                perror("setsockopt(SO_NOSIGPIPE client)");
                // not fatal, keep the client if you like
                           }
        }
#endif

        if (srv->num_clients >= NMMO_MAX_CLIENTS) {
            printf("[nmmo-server] rejecting client (max clients)\n");
            close(cfd);
            continue;
        }

        char ipbuf[64];
        snprintf(ipbuf, sizeof(ipbuf), "%s", inet_ntoa(cli_addr.sin_addr));
        printf("[nmmo-server] client connected %s:%d (fd=%d)\n",
               ipbuf, ntohs(cli_addr.sin_port), cfd);

        if (nmmo_send_map_v1(cfd, env) < 0) {
            printf("[nmmo-server] failed to send map to client\n");
            close(cfd);
            continue;
        }
        // send current tick snapshot so the client can render immediately
        if (nmmo_send_tick_v1(cfd, env) < 0) {
            // ok to ignore; client can wait for next broadcast
        }

        srv->client_fds[srv->num_clients++] = cfd;
    }
}

static void nmmo_server_poll_accept(NMMOServer *srv, const MMO *env) {
    if (srv->listen_fd < 0) return;

    for (;;) {
        struct sockaddr_in cli;
        socklen_t len = sizeof(cli);
        int cfd = accept(srv->listen_fd, (struct sockaddr *) &cli, &len);
        if (cfd < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) break;
            if (errno == EINTR) continue;
            perror("accept");
            break;
        }

        if (srv->num_clients >= NMMO_MAX_CLIENTS) {
            fprintf(stderr, "[nmmo-server] reject: max clients\n");
            close(cfd);
            continue;
        }

        #ifdef SO_NOSIGPIPE
        int yes = 1;
        if (setsockopt(cfd, SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof(yes)) < 0) {
            perror("setsockopt(SO_NOSIGPIPE client)");
        }
        #endif

        if (nmmo_set_nonblocking(cfd) < 0) {
            perror("fcntl(client_fd)");
            close(cfd);
            continue;
        }

        // Send initial map snapshot (header+terrain)
        if (nmmo_send_map_v1(cfd, env) < 0) {
            fprintf(stderr, "[nmmo-server] failed to send map to new client\n");
            close(cfd);
            continue;
        }

        srv->client_fds[srv->num_clients++] = cfd;
        printf("[nmmo-server] client connected (%d/%d)\n", srv->num_clients, NMMO_MAX_CLIENTS);
    }
}

static void nmmo_server_broadcast_tick(NMMOServer *srv, const MMO *env) {
    if (srv->num_clients <= 0) return;

    // Serialize once
    static uint8_t tickbuf[1 << 20];
    size_t n = nmmo_serialize_tick_v1(env, tickbuf, sizeof(tickbuf));
    if (n == 0) return;

    // Send to all, compact list on disconnect
    int out = 0;
    for (int i = 0; i < srv->num_clients; i++) {
        int fd = srv->client_fds[i];
        int rc = nmmo_send_all_nonblocking(fd, tickbuf, n);
        if (rc < 0) {
            close(fd);
            continue;
        }
        srv->client_fds[out++] = fd;
    }
    if (out != srv->num_clients) {
        printf("[nmmo-server] clients: %d -> %d\n", srv->num_clients, out);
        srv->num_clients = out;
    }
}

static void nmmo_server_shutdown(NMMOServer *srv) {
    for (int i = 0; i < srv->num_clients; i++) close(srv->client_fds[i]);
    srv->num_clients = 0;
    if (srv->listen_fd >= 0) close(srv->listen_fd);
    srv->listen_fd = -1;
}
#endif // !PLATFORM_WEB

// -------------------------------------------------------------------------
// Demo / MMO stepping glue
// -------------------------------------------------------------------------

typedef struct {
    MMO env;
    Client *client;
    char *terrain;
    NMMOServer server;
#if defined(PLATFORM_WEB)
    int web_dummy;
#endif
} DemoCtx;

static void step_and_maybe_send(DemoCtx *ctx) {
    // Advance game (original behavior)
    ctx->env.tick++;

#if !defined(PLATFORM_WEB)
    // accept + broadcast only in native server mode
    nmmo_server_poll_accept(&ctx->server, &ctx->env);
    nmmo_server_broadcast_tick(&ctx->server, &ctx->env);
#endif

    // Render
    (void) c_render(&ctx->env);
}


void demo(int num_players) {
//    srand(time(NULL));
//    Weights* weights = load_weights("resources/nmmo3/nmmo3_weights.bin", 3387547);
//    MMONet* net = init_mmonet(weights, num_players);

 	// Make all send()/write() calls fail with EPIPE instead of killing the process
    signal(SIGPIPE, SIG_IGN);

    MMO env = {
        .client = NULL,
        .width = 30,
        .height = 30,
        .num_players = num_players,
        .num_enemies = 0,
        .num_resources = 0,
        .num_weapons = 0,
        .num_gems = 0,
        .tiers = 5,
        .levels = 40,
        .teleportitis_prob = 0.0,
        .enemy_respawn_ticks = 2,
        .item_respawn_ticks = 100,
        .x_window = 7,
        .y_window = 5,
    };
    allocate_mmo(&env);

    c_reset(&env);
    c_render(&env);

#if !defined(PLATFORM_WEB)
    // Start the realtime snapshot server (clients connect + render locally)
    NMMOServer srv;
    if (nmmo_server_init(&srv, NMMO_SERVER_PORT) != 0) {
        // continue without server
        memset(&srv, 0, sizeof(srv));
        srv.listen_fd = -1;
    }
#endif

    int human_action = ATN_NOOP;
    bool human_mode = true;
    int i = 1;
    while (!WindowShouldClose()) {
#if !defined(PLATFORM_WEB)
        nmmo_server_accept_new(&srv, &env);
#endif
        if (IsKeyPressed(KEY_LEFT_CONTROL)) {
            human_mode = !human_mode;
        }
        if (i % 36 == 0) {
//            forward(net, env.observations, env.actions);
            if (human_mode) {
                env.actions[0] = human_action;
            }

            c_step(&env);
#if !defined(PLATFORM_WEB)
            nmmo_server_broadcast_tick(&srv, &env);
#endif
            human_action = ATN_NOOP;
        }
        int atn = c_render(&env);
        if (atn != ATN_NOOP) {
            human_action = atn;
        }
        i = (i + 1) % 36;
    }

    //    free_mmonet(net);
    //    free(weights);
#if !defined(PLATFORM_WEB)
    nmmo_server_shutdown(&srv);
#endif

    free_allocated_mmo(&env);
    //close_client(client);
}

// -------------------------------------------------------------------------
// Viewer tick parsing helpers
// -------------------------------------------------------------------------


static void rebuild_pids(MMO *env) {
    int W = env->width;
    int H = env->height;
    int N = W * H;
    for (int i = 0; i < N; i++) env->pids[i] = -1;

    for (int i = 0; i < env->num_players; i++) {
        Entity *e = &env->players[i];
        if (e->hp <= 0) continue;
        int r = e->r, c = e->c;
        if ((unsigned) r < (unsigned) H && (unsigned) c < (unsigned) W) {
            env->pids[r * W + c] = (short) i;
        }
    }
    for (int i = 0; i < env->num_enemies; i++) {
        Entity *e = &env->enemies[i];
        if (e->hp <= 0) continue;
        int r = e->r, c = e->c;
        if ((unsigned) r < (unsigned) H && (unsigned) c < (unsigned) W) {
            env->pids[r * W + c] = (short) (env->num_players + i);
        }
    }
}

static void apply_tick_entities(MMO *env, const NetEntityV1 *ents, uint32_t n_entities) {
    // Clamp to local allocations
    uint32_t max = (uint32_t) (env->num_players + env->num_enemies);
    if (n_entities > max) n_entities = max;

    for (uint32_t i = 0; i < n_entities; i++) {
        int id = (int) ntohl(ents[i].id);
        int type = (int) ntohl(ents[i].type);
        int r = (int) ntohl(ents[i].r);
        int c = (int) ntohl(ents[i].c);
        int hp = (int) ntohl(ents[i].hp);
        int hp_max = (int) ntohl(ents[i].hp_max);
        int comb_lvl = (int) ntohl(ents[i].comb_lvl);
        int element = (int) ntohl(ents[i].element);
        int anim = (int) ntohl(ents[i].anim);
        int dir = (int) ntohl(ents[i].dir);

        // Players first [0..num_players)
        if (id >= 0 && id < env->num_players) {
            Entity *e = &env->players[id];
            e->type = type;
            e->r = r;
            e->c = c;
            e->hp = hp;
            e->hp_max = hp_max;
            e->comb_lvl = comb_lvl;
            e->element = element;
            e->anim = anim;
            e->dir = dir;
            continue;
        }

        // Enemies [num_players..num_players+num_enemies)
        int eid = id - env->num_players;
        if (eid >= 0 && eid < env->num_enemies) {
            Entity *e = &env->enemies[eid];
            e->type = type;
            e->r = r;
            e->c = c;
            e->hp = hp;
            e->hp_max = hp_max;
            e->comb_lvl = comb_lvl;
            e->element = element;
            e->anim = anim;
            e->dir = dir;
            continue;
        }
    }

    // rebuild pid occupancy after applying latest tick
    // (existing helper)
    // NOTE: rebuild_pids is defined below in this file.
    // It is unchanged.
    // (call left as-is)
    // If rebuild_pids appears earlier in your file, this still compiles due to C's single pass w/ prototypes
    // but the original file already had it wired this way.
    // Keep behavior identical:
    rebuild_pids(env);
}
/* Parse at most ONE complete tick message from rxbuf.
   Each call advances the env by at most one tick, so we don't "fast-forward"
   through bursts of ticks in a single render frame. */
static void parse_ticks_into_env(MMO *env, uint8_t *rxbuf, size_t *rxlen_io) {
    size_t rxlen = *rxlen_io;

    // Need at least a header
    if (rxlen < sizeof(NetTickHeaderV1)) {
        return;
    }

    NetTickHeaderV1 hdr;
    memcpy(&hdr, rxbuf, sizeof(hdr));
    uint32_t tick       = ntohl(hdr.tick);
    uint32_t n_entities = ntohl(hdr.n_entities);

    size_t need = sizeof(NetTickHeaderV1) + (size_t) n_entities * sizeof(NetEntityV1);
    if (rxlen < need) {
        // Not enough bytes for the full tick yet: wait for more data
        return;
    }

    const NetEntityV1 *ents = (const NetEntityV1 *) (rxbuf + sizeof(NetTickHeaderV1));

    // Optional: ignore out-of-order / duplicate ticks
    if (tick > (uint32_t) env->tick) {
        env->tick = (int) tick;
        apply_tick_entities(env, ents, n_entities);
    }

    // Consume exactly one tick
    size_t remain = rxlen - need;
    if (remain > 0) {
        memmove(rxbuf, rxbuf + need, remain);
    }
    *rxlen_io = remain;
}

#define DEFAULT_HOST "app.microbrew.ai"
#define DEFAULT_PORT "443"

/* A small ring-ish buffer for TCP stream parsing */
#define RX_CAP (1u << 20) /* 1 MiB should be plenty for small maps */

#if defined(PLATFORM_WEB)

typedef enum {
    VIEWER_WS_WAIT_HEADER = 0,
    VIEWER_WS_WAIT_TERRAIN = 1,
    VIEWER_WS_RUNNING = 2,
    VIEWER_WS_FAILED = 3
} ViewerWSState;

typedef struct {
    EMSCRIPTEN_WEBSOCKET_T ws;

    uint8_t *rxbuf;
    size_t rxlen;
    size_t rxcap;

    ViewerWSState state;

    // Map/bootstrap state
    NetMapHeaderV1 mh;
    int width, height, num_players, num_enemies;
    size_t terrain_bytes;
    char *terrain;

    // Render env (same as TCP path)
    MMO env;
    int env_ready;
} WebViewer;

static void ws_ensure_cap(WebViewer *v, size_t add) {
    if (v->rxlen + add <= v->rxcap) return;
    size_t newcap = v->rxcap ? v->rxcap : RX_CAP;
    while (newcap < v->rxlen + add) newcap *= 2u;
    uint8_t *p = (uint8_t *) realloc(v->rxbuf, newcap);
    if (!p) {
        fprintf(stderr, "[viewer] OOM rxbuf grow\n");
        v->state = VIEWER_WS_FAILED;
        return;
    }
    v->rxbuf = p;
    v->rxcap = newcap;
}

static void ws_consume(WebViewer *v, size_t n) {
    if (n == 0 || n > v->rxlen) return;
    size_t remain = v->rxlen - n;
    if (remain) memmove(v->rxbuf, v->rxbuf + n, remain);
    v->rxlen = remain;
}

static void ws_try_init_env(WebViewer *v) {
    if (v->state == VIEWER_WS_FAILED || v->env_ready) return;

    // Need header first
    if (v->state == VIEWER_WS_WAIT_HEADER) {
        if (v->rxlen < sizeof(NetMapHeaderV1)) return;
        memcpy(&v->mh, v->rxbuf, sizeof(NetMapHeaderV1));
        ws_consume(v, sizeof(NetMapHeaderV1));

        uint32_t magic = ntohl(v->mh.magic);
        uint16_t version = ntohs(v->mh.version);
        if (magic != NMMO_MAGIC || version != 1) {
            fprintf(stderr, "[viewer] bad protocol: magic=0x%08x version=%u\n",
                    (unsigned) magic, (unsigned) version);
            v->state = VIEWER_WS_FAILED;
            return;
        }

        v->width = (int) ntohl(v->mh.width);
        v->height = (int) ntohl(v->mh.height);
        v->num_players = (int) ntohl(v->mh.num_players);
        v->num_enemies = (int) ntohl(v->mh.num_enemies);

        if (v->width <= 0 || v->height <= 0 || v->width > 8192 || v->height > 8192) {
            fprintf(stderr, "[viewer] invalid map dims %d x %d\n", v->width, v->height);
            v->state = VIEWER_WS_FAILED;
            return;
        }
        if (v->num_players < 0 || v->num_players > 4096 || v->num_enemies < 0 || v->num_enemies > 4096) {
            fprintf(stderr, "[viewer] invalid entity counts players=%d enemies=%d\n",
                    v->num_players, v->num_enemies);
            v->state = VIEWER_WS_FAILED;
            return;
        }

        v->terrain_bytes = (size_t) v->width * (size_t) v->height;
        v->terrain = (char *) malloc(v->terrain_bytes);
        if (!v->terrain) {
            fprintf(stderr, "[viewer] OOM terrain\n");
            v->state = VIEWER_WS_FAILED;
            return;
        }

        v->state = VIEWER_WS_WAIT_TERRAIN;
    }

    // Then terrain
    if (v->state == VIEWER_WS_WAIT_TERRAIN) {
        if (v->rxlen < v->terrain_bytes) return;
        memcpy(v->terrain, v->rxbuf, v->terrain_bytes);
        ws_consume(v, v->terrain_bytes);

        // Build a minimal MMO env for rendering (same allocations as TCP path)
        memset(&v->env, 0, sizeof(v->env));
        v->env.client = NULL;
        v->env.width = v->width;
        v->env.height = v->height;
        v->env.num_players = v->num_players;
        v->env.num_enemies = v->num_enemies;
        v->env.terrain = v->terrain;

        v->env.players = (Entity *) calloc((size_t) v->num_players, sizeof(Entity));
        v->env.enemies = (Entity *) calloc((size_t) v->num_enemies, sizeof(Entity));
        v->env.pids = (short *) malloc((size_t) v->width * (size_t) v->height * sizeof(short));
        v->env.items = (unsigned char *) calloc((size_t) v->width * (size_t) v->height, sizeof(unsigned char));
        v->env.actions = (int *) calloc((size_t) (v->num_players > 0 ? v->num_players : 1), sizeof(int)); // safe

        if (!v->env.players || !v->env.enemies || !v->env.pids || !v->env.items || !v->env.actions) {
            fprintf(stderr, "[viewer] OOM allocations\n");
            v->state = VIEWER_WS_FAILED;
            return;
        }

        for (int i = 0; i < v->width * v->height; i++) v->env.pids[i] = -1;

        v->env_ready = 1;
        v->state = VIEWER_WS_RUNNING;

        // Kick one render to get the window initialized (matches existing behavior of calling c_render repeatedly)
        (void) c_render(&v->env);
    }
}

static EM_BOOL ws_onopen(int eventType, const EmscriptenWebSocketOpenEvent *e, void *userData) {
    (void) eventType; (void) e;
    WebViewer *v = (WebViewer *) userData;
    printf("[viewer] websocket connected\n");
    v->state = VIEWER_WS_WAIT_HEADER;
    return EM_TRUE;
}

static EM_BOOL ws_onerror(int eventType, const EmscriptenWebSocketErrorEvent *e, void *userData) {
    (void) eventType; (void) e;
    WebViewer *v = (WebViewer *) userData;
    fprintf(stderr, "[viewer] websocket error\n");
    v->state = VIEWER_WS_FAILED;
    return EM_TRUE;
}

static EM_BOOL ws_onclose(int eventType, const EmscriptenWebSocketCloseEvent *e, void *userData) {
    (void) eventType; (void) e;
    WebViewer *v = (WebViewer *) userData;
    fprintf(stderr, "[viewer] websocket closed\n");
    v->state = VIEWER_WS_FAILED;
    return EM_TRUE;
}

static EM_BOOL ws_onmessage(int eventType, const EmscriptenWebSocketMessageEvent *e, void *userData) {
    (void) eventType;
    WebViewer *v = (WebViewer *) userData;
    if (v->state == VIEWER_WS_FAILED) return EM_TRUE;
    if (e->isText) return EM_TRUE; // protocol is binary

    ws_ensure_cap(v, (size_t) e->numBytes);
    if (v->state == VIEWER_WS_FAILED) return EM_TRUE;

    memcpy(v->rxbuf + v->rxlen, e->data, (size_t) e->numBytes);
    v->rxlen += (size_t) e->numBytes;

    // If header/terrain arrive, bootstrap env as soon as possible
    ws_try_init_env(v);
    return EM_TRUE;
}

static void ws_main_loop(void *arg) {
    WebViewer *v = (WebViewer *) arg;
    if (v->state == VIEWER_WS_FAILED) return;

    // If env not ready yet, we only render once it is.
    if (!v->env_ready) return;

    // Same parsing + rendering as TCP loop (just no recv())
    parse_ticks_into_env(&v->env, v->rxbuf, &v->rxlen);

    // Render using existing code (local UI keys like H work here)
    (void) c_render(&v->env);
}

#endif // PLATFORM_WEB


static int set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags == -1) return -1;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

static int recv_all_blocking(int fd, void *buf, size_t len) {
    uint8_t *p = (uint8_t *) buf;
    while (len > 0) {
        ssize_t n = recv(fd, p, len, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            perror("recv");
            return -1;
        }
        if (n == 0) return -1;
        p += (size_t) n;
        len -= (size_t) n;
    }
    return 0;
}

static int connect_tcp(const char *host, const char *port) {
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    struct addrinfo *res = NULL;
    int rc = getaddrinfo(host, port, &hints, &res);
    if (rc != 0) {
        fprintf(stderr, "getaddrinfo(%s,%s): %s\n", host, port, gai_strerror(rc));
        return -1;
    }

    int fd = -1;
    for (struct addrinfo *it = res; it != NULL; it = it->ai_next) {
        fd = socket(it->ai_family, it->ai_socktype, it->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, it->ai_addr, it->ai_addrlen) == 0) {
            break;
        }
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);

    if (fd < 0) {
        fprintf(stderr, "Failed to connect to %s:%s\n", host, port);
        return -1;
    }
    return fd;
}


int main2(int argc, char **argv) {
    const char *host = DEFAULT_HOST;
    const char *port = DEFAULT_PORT;


#if defined(PLATFORM_WEB)
    // if (!emscripten_websocket_is_supported()) {
    //     fprintf(stderr, "[viewer] websockets not supported\n");
    //     return 1;
    // }

    WebViewer *v = (WebViewer *) calloc(1, sizeof(WebViewer));
    if (!v) return 1;

    v->rxcap = RX_CAP;
    v->rxbuf = (uint8_t *) malloc(v->rxcap);
    if (!v->rxbuf) return 1;

    v->state = VIEWER_WS_WAIT_HEADER;

    char url[256];
    snprintf(url, sizeof(url), "wss://%s:%s", host, port);

    EmscriptenWebSocketCreateAttributes attr;
    emscripten_websocket_init_create_attributes(&attr);
    attr.url = url;
    attr.createOnMainThread = 1;

    v->ws = emscripten_websocket_new(&attr);
    if (v->ws <= 0) {
        fprintf(stderr, "[viewer] failed to create websocket to %s\n", url);
        return 1;
    }

    // Ensure e->data points to bytes (not Blob)
    // emscripten_websocket_set_binary_type(v->ws, EMSCRIPTEN_WEBSOCKET_BINARY_TYPE_ARRAYBUFFER);

    emscripten_websocket_set_onopen_callback(v->ws, v, ws_onopen);
    emscripten_websocket_set_onerror_callback(v->ws, v, ws_onerror);
    emscripten_websocket_set_onclose_callback(v->ws, v, ws_onclose);
    emscripten_websocket_set_onmessage_callback(v->ws, v, ws_onmessage);

    printf("[viewer] connecting to %s\n", url);

    emscripten_set_main_loop_arg(ws_main_loop, v, 0, 1);
    return 0;
#else

    int fd = connect_tcp(host, port);
    if (fd < 0) return 1;

    printf("[viewer] connected to %s:%s\n", host, port);

    // Read map header (blocking)
    NetMapHeaderV1 mh;
    if (recv_all_blocking(fd, &mh, sizeof(mh)) != 0) {
        fprintf(stderr, "[viewer] failed to read map header\n");
        close(fd);
        return 1;
    }

    uint32_t magic = ntohl(mh.magic);
    uint16_t version = ntohs(mh.version);
    if (magic != NMMO_MAGIC || version != 1) {
        fprintf(stderr, "[viewer] bad protocol: magic=0x%08x version=%u\n",
                (unsigned) magic, (unsigned) version);
        close(fd);
        return 1;
    }

    int width = (int) ntohl(mh.width);
    int height = (int) ntohl(mh.height);
    int num_players = (int) ntohl(mh.num_players);
    int num_enemies = (int) ntohl(mh.num_enemies);

    if (width <= 0 || height <= 0 || width > 8192 || height > 8192) {
        fprintf(stderr, "[viewer] invalid map dims %d x %d\n", width, height);
        close(fd);
        return 1;
    }
    if (num_players < 0 || num_players > 4096 || num_enemies < 0 || num_enemies > 4096) {
        fprintf(stderr, "[viewer] invalid entity counts players=%d enemies=%d\n", num_players, num_enemies);
        close(fd);
        return 1;
    }

    size_t terrain_bytes = (size_t) width * (size_t) height;
    char *terrain = (char *) malloc(terrain_bytes);
    if (!terrain) {
        fprintf(stderr, "[viewer] OOM terrain\n");
        close(fd);
        return 1;
    }

    if (recv_all_blocking(fd, terrain, terrain_bytes) != 0) {
        fprintf(stderr, "[viewer] failed to read terrain\n");
        free(terrain);
        close(fd);
        return 1;
    }

    // Switch to nonblocking for the render loop
    if (set_nonblocking(fd) != 0) {
        perror("fcntl(nonblocking)");
        // not fatal, but the render loop may hitch
    }

    // Build a minimal MMO env for rendering
    MMO env;
    memset(&env, 0, sizeof(env));
    env.client = NULL;
    env.width = width;
    env.height = height;
    env.num_players = num_players;
    env.num_enemies = num_enemies;
    env.terrain = terrain;

    env.players = (Entity *) calloc((size_t) num_players, sizeof(Entity));
    env.enemies = (Entity *) calloc((size_t) num_enemies, sizeof(Entity));
    env.pids = (short *) malloc((size_t) width * (size_t) height * sizeof(short));
    env.items = (unsigned char *) calloc((size_t) width * (size_t) height, sizeof(unsigned char));
    env.actions = (int *) calloc((size_t) (num_players > 0 ? num_players : 1), sizeof(int)); // safe

    if (!env.players || !env.enemies || !env.pids || !env.items || !env.actions) {
        fprintf(stderr, "[viewer] OOM allocations\n");
        close(fd);
        free(terrain);
        free(env.players);
        free(env.enemies);
        free(env.pids);
        free(env.items);
        free(env.actions);
        return 1;
    }

    for (int i = 0; i < width * height; i++) env.pids[i] = -1;

    // Stream buffer
    uint8_t *rxbuf = (uint8_t *) malloc(RX_CAP);
    size_t rxlen = 0;
    if (!rxbuf) {
        fprintf(stderr, "[viewer] OOM rxbuf\n");
        close(fd);
        free(terrain);
        free(env.players);
        free(env.enemies);
        free(env.pids);
        free(env.items);
        free(env.actions);
        return 1;
    }

    // Render loop: poll network, update env, render
    while (true) {
        // Drain socket into rxbuf
        for (;;) {
            if (rxlen >= RX_CAP) {
                // buffer full; drop everything (shouldn't happen with reasonable tick sizes)
                rxlen = 0;
            }
            ssize_t n = recv(fd, rxbuf + rxlen, RX_CAP - rxlen, 0);
            if (n > 0) {
                rxlen += (size_t) n;
                continue;
            }
            if (n == 0) {
                fprintf(stderr, "[viewer] server closed connection\n");
                goto cleanup;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                break;
            }
            if (errno == EINTR) continue;
            perror("recv");
            goto cleanup;
        }

        parse_ticks_into_env(&env, rxbuf, &rxlen);

        // Render using existing code (local UI keys like H work here)
        (void) c_render(&env);
    }

cleanup:
    if (env.client) {
        close_client(env.client);
        env.client = NULL;
    }
    CloseWindow();

    close(fd);
    free(rxbuf);

    free(env.players);
    free(env.enemies);
    free(env.pids);
    free(env.items);
    free(env.actions);

    free(terrain);
    return 0;

#endif
}

static void ignore_sigpipe(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = SIG_IGN;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGPIPE, &sa, NULL);
}

int main() {
    signal(SIGPIPE, SIG_IGN);
    ignore_sigpipe();

    const char *render_type = getenv("RENDER_TYPE");
    if (render_type != NULL && strcmp(render_type, "server") == 0) {
        demo(2);
    } else {
        main2(0, NULL);
    }
    return 0;
}
