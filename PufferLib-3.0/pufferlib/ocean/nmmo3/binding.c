#include <Python.h>
#include "nmmo3.h"

#define Env MMO

#define NMMO_MAGIC 0x4E4D4D33u

// Forward declarations for our extra Python-callable functions
static PyObject* hello_world(PyObject* self, PyObject* args);
static PyObject* py_net_map_v1(PyObject* self, PyObject* args);
static PyObject* py_net_tick_v1(PyObject* self, PyObject* args);

// These get spliced into env_binding.h's method table.
#define MY_METHODS                                           \
    {"hello_world", (PyCFunction)hello_world, METH_NOARGS,   \
        "Return 'hello world'."},                            \
    {"net_map_v1", (PyCFunction)py_net_map_v1, METH_VARARGS, \
        "Return NetMapHeaderV1-style header + terrain."},    \
    {"net_tick_v1", (PyCFunction)py_net_tick_v1, METH_VARARGS,\
        "Return NetTickHeaderV1-style header + entities."}

#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->width = unpack(kwargs, "width");
    env->height = unpack(kwargs, "height");
    env->num_players = unpack(kwargs, "num_players");
    env->num_enemies = unpack(kwargs, "num_enemies");
    env->num_resources = unpack(kwargs, "num_resources");
    env->num_weapons = unpack(kwargs, "num_weapons");
    env->num_gems = unpack(kwargs, "num_gems");
    env->tiers = unpack(kwargs, "tiers");
    env->levels = unpack(kwargs, "levels");
    env->teleportitis_prob = unpack(kwargs, "teleportitis_prob");
    env->enemy_respawn_ticks = unpack(kwargs, "enemy_respawn_ticks");
    env->item_respawn_ticks = unpack(kwargs, "item_respawn_ticks");
    env->x_window = unpack(kwargs, "x_window");
    env->y_window = unpack(kwargs, "y_window");
    env->reward_combat_level = unpack(kwargs, "reward_combat_level");
    env->reward_prof_level = unpack(kwargs, "reward_prof_level");
    env->reward_item_level = unpack(kwargs, "reward_item_level");
    env->reward_market = unpack(kwargs, "reward_market");
    env->reward_death = unpack(kwargs, "reward_death");
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "return_comb_lvl", log->return_comb_lvl);
    assign_to_dict(dict, "return_prof_lvl", log->return_prof_lvl);
    assign_to_dict(dict, "return_item_atk_lvl", log->return_item_atk_lvl);
    assign_to_dict(dict, "return_item_def_lvl", log->return_item_def_lvl);
    assign_to_dict(dict, "return_market_buy", log->return_market_buy);
    assign_to_dict(dict, "return_market_sell", log->return_market_sell);
    assign_to_dict(dict, "return_death", log->return_death);
    assign_to_dict(dict, "min_comb_prof", log->min_comb_prof);
    assign_to_dict(dict, "purchases", log->purchases);
    assign_to_dict(dict, "sales", log->sales);
    assign_to_dict(dict, "equip_attack", log->equip_attack);
    assign_to_dict(dict, "equip_defense", log->equip_defense);
    assign_to_dict(dict, "r", log->r);
    assign_to_dict(dict, "c", log->c);
    return 0;
}

// Simple hello_world for sanity
static PyObject* hello_world(PyObject* self, PyObject* args) {
    (void)self; (void)args;
    return PyUnicode_FromString("hello world");
}

// net_map_v1(env_handle: int) -> (header_dict, terrain_bytes)
static PyObject* py_net_map_v1(PyObject* self, PyObject* args) {
    (void)self;

    Env* env = unpack_env(args);
    if (!env) {
        // unpack_env sets a Python exception on failure
        return NULL;
    }

    // Header: mirrors NetMapHeaderV1 fields, but in host order.
    PyObject* header = Py_BuildValue(
        "{sI sI sI sI sI sI sI}",
        "magic",       (unsigned int)NMMO_MAGIC,
        "version",     (unsigned int)1,
        "reserved",    (unsigned int)0,
        "width",       env->width,
        "height",      env->height,
        "num_players", env->num_players,
        "num_enemies", env->num_enemies
    );
    if (!header) {
        return NULL;
    }

    // Terrain is width * height bytes
    Py_ssize_t terrain_bytes =
        (Py_ssize_t)((size_t)env->width * (size_t)env->height);
    PyObject* terrain = PyBytes_FromStringAndSize(
        (const char*)env->terrain, terrain_bytes
    );
    if (!terrain) {
        Py_DECREF(header);
        return NULL;
    }

    PyObject* result = PyTuple_New(2);
    if (!result) {
        Py_DECREF(header);
        Py_DECREF(terrain);
        return NULL;
    }

    PyTuple_SET_ITEM(result, 0, header);   // steals ref
    PyTuple_SET_ITEM(result, 1, terrain);  // steals ref
    return result;
}

// net_tick_v1(env_handle: int) -> (header_dict, [entity_dict...])
static PyObject* py_net_tick_v1(PyObject* self, PyObject* args) {
    (void)self;

    Env* env = unpack_env(args);
    if (!env) {
        return NULL;
    }

    int total = env->num_players + env->num_enemies;

    PyObject* header = Py_BuildValue(
        "{sI sI}",
        "tick",       (unsigned int)env->tick,
        "n_entities", (unsigned int)total
    );
    if (!header) {
        return NULL;
    }

    PyObject* ents = PyList_New(total);
    if (!ents) {
        Py_DECREF(header);
        return NULL;
    }

    int idx = 0;

    // Players: pid == index
    for (int i = 0; i < env->num_players; ++i, ++idx) {
        const Entity* e = &env->players[i];
        PyObject* d = Py_BuildValue(
            "{sI sI sI sI sI sI sI sI sI sI}",
            "id",       (unsigned int)i,
            "type",     e->type,
            "r",        e->r,
            "c",        e->c,
            "hp",       e->hp,
            "hp_max",   e->hp_max,
            "comb_lvl", e->comb_lvl,
            "element",  e->element,
            "anim",     e->anim,
            "dir",      e->dir
        );
        if (!d) {
            Py_DECREF(header);
            Py_DECREF(ents);
            return NULL;
        }
        PyList_SET_ITEM(ents, idx, d);  // steals ref
    }

    // Enemies: pid == num_players + index
    for (int i = 0; i < env->num_enemies; ++i, ++idx) {
        const Entity* e = &env->enemies[i];
        int id = env->num_players + i;
        PyObject* d = Py_BuildValue(
            "{sI sI sI sI sI sI sI sI sI sI}",
            "id",       (unsigned int)id,
            "type",     e->type,
            "r",        e->r,
            "c",        e->c,
            "hp",       e->hp,
            "hp_max",   e->hp_max,
            "comb_lvl", e->comb_lvl,
            "element",  e->element,
            "anim",     e->anim,
            "dir",      e->dir
        );
        if (!d) {
            Py_DECREF(header);
            Py_DECREF(ents);
            return NULL;
        }
        PyList_SET_ITEM(ents, idx, d);
    }

    PyObject* result = PyTuple_New(2);
    if (!result) {
        Py_DECREF(header);
        Py_DECREF(ents);
        return NULL;
    }

    PyTuple_SET_ITEM(result, 0, header);
    PyTuple_SET_ITEM(result, 1, ents);
    return result;
}
