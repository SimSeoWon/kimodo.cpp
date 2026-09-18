// Command-line bridge for the localhost demo.  It deliberately uses only the
// public C++ model API, so the demo exercises the same text route as embedders.
#include <kimodo/kimodo.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include <sstream>

namespace {
void write_f32(const std::filesystem::path &path, const std::vector<float> &values) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open " + path.string());
    out.write(reinterpret_cast<const char *>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!out) throw std::runtime_error("cannot write " + path.string());
}

void write_motion(const std::filesystem::path &output, const kimodo::motion_data &motion) {
    std::filesystem::create_directories(output);
    write_f32(output / "root_positions.f32", motion.root_positions);
    write_f32(output / "local_rotations_xyzw.f32", motion.local_rotations_xyzw);
}

std::vector<std::string> split_fields(const std::string &line) {
    std::vector<std::string> fields;
    std::string field;
    std::istringstream input(line);
    while (std::getline(input, field, '\t')) fields.push_back(field);
    return fields;
}

std::string protocol_error(std::string message) {
    for (char &c : message) if (c == '\n' || c == '\r' || c == '\t') c = ' ';
    return message;
}

// CFG weights were hardcoded (2.0, 2.0) here; env vars let the web UI expose them as
// sliders without a new CLI flag (same pattern as KIMODO_BACKEND/KIMODO_TEXT_LAYER_CHUNK
// in the library). Unset keeps the exact prior default.
float env_float(const char *name, float fallback) {
    const char *value = std::getenv(name);
    if (!value) return fallback;
    char *end = nullptr;
    const float parsed = std::strtof(value, &end);
    if (end == value || *end != '\0')
        throw std::runtime_error(std::string(name) + " must be a number");
    return parsed;
}

std::string read_file(const char *path) {
    std::ifstream in(path);
    const std::string text{std::istreambuf_iterator<char>(in), {}};
    if (!in && text.empty()) throw std::runtime_error(std::string("cannot read ") + path);
    return text;
}

std::vector<kimodo::pose_constraint> read_constraints(const char *path) {
    std::ifstream in(path);if(!in)throw std::runtime_error(std::string("cannot read constraints ")+path);
    std::vector<kimodo::pose_constraint> result;std::string line;unsigned line_number=0;
    while(std::getline(in,line)){++line_number;if(line.empty()||line[0]=='#')continue;std::istringstream row(line);kimodo::pose_constraint value;int position=0,rotation=0;
        if(!(row>>value.frame>>value.joint>>position>>value.world_position[0]>>value.world_position[1]>>value.world_position[2]
             >>rotation>>value.world_rotation_xyzw[0]>>value.world_rotation_xyzw[1]>>value.world_rotation_xyzw[2]>>value.world_rotation_xyzw[3]))
            throw std::runtime_error("invalid constraint line "+std::to_string(line_number));
        value.constrain_position=position!=0;value.constrain_rotation=rotation!=0;result.push_back(value);}
    return result;
}
} // namespace

int main(int argc, char **argv) try {
    if (argc == 4 && std::string_view(argv[1]) == "--server") {
        auto model = kimodo::model::load(argv[2], argv[3]);
        if (!model) throw std::runtime_error(model.error());
        std::string line;
        while (std::getline(std::cin, line)) {
            try {
                const auto fields = split_fields(line);
                if (fields.size() < 6 || (fields.size() - 4) % 2 != 0)
                    throw std::runtime_error("invalid server request");
                const auto transition = static_cast<unsigned>(std::stoul(fields[0]));
                const auto steps = static_cast<unsigned>(std::stoul(fields[1]));
                const auto seed = static_cast<std::uint64_t>(std::stoull(fields[2]));
                std::vector<kimodo::prompt_segment> segments;
                for (size_t index = 4; index < fields.size(); index += 2) {
                    std::ifstream prompt_file(fields[index + 1]);
                    const std::string prompt{std::istreambuf_iterator<char>(prompt_file), {}};
                    if (!prompt_file && prompt.empty()) throw std::runtime_error("cannot read sequence prompt");
                    segments.push_back({prompt, static_cast<unsigned>(std::stoul(fields[index]))});
                }
                auto motion = (*model)->generate_text_sequence(segments, transition, steps, seed, 2.F, 2.F);
                if (!motion) throw std::runtime_error(motion.error());
                write_motion(fields[3], *motion);
                std::cout << "OK\t" << motion->frames << '\t' << motion->joints << '\n' << std::flush;
            } catch (const std::exception &error) {
                std::cout << "ERR\t" << protocol_error(error.what()) << '\n' << std::flush;
            }
        }
        return 0;
    }
    if (argc >= 10 && std::string_view(argv[3]) == "--sequence") {
        if ((argc - 8) % 2 != 0) throw std::runtime_error("sequence requires FRAME PROMPT.txt pairs");
        const auto transition = static_cast<unsigned>(std::stoul(argv[4]));
        const auto steps = static_cast<unsigned>(std::stoul(argv[5]));
        const auto seed = static_cast<std::uint64_t>(std::stoull(argv[6]));
        std::vector<kimodo::prompt_segment> segments;
        for (int index=8; index<argc; index+=2) {
            std::ifstream prompt_file(argv[index+1]);
            const std::string prompt{std::istreambuf_iterator<char>(prompt_file), {}};
            if (!prompt_file && prompt.empty()) throw std::runtime_error("cannot read sequence prompt");
            segments.push_back({prompt, static_cast<unsigned>(std::stoul(argv[index]))});
        }
        auto model = kimodo::model::load(argv[1], argv[2]);
        if (!model) throw std::runtime_error(model.error());
        const char *constraint_path=std::getenv("KIMODO_CONSTRAINTS_FILE");
        auto constraints=constraint_path?read_constraints(constraint_path):std::vector<kimodo::pose_constraint>{};
        auto motion = constraints.empty()
            ? (*model)->generate_text_sequence(segments, transition, steps, seed,
                env_float("KIMODO_TEXT_CFG", 2.F), env_float("KIMODO_CONSTRAINT_CFG", 2.F))
            : (*model)->generate_text_sequence_constrained(segments, transition, steps, seed,
                env_float("KIMODO_TEXT_CFG", 2.F), env_float("KIMODO_CONSTRAINT_CFG", 2.F),constraints);
        if (!motion) throw std::runtime_error(motion.error());
        write_motion(argv[7], *motion);
        std::cout << "generated " << motion->frames << " frames with " << motion->joints << " joints\n";
        return 0;
    }
    if (argc != 8 && argc != 9) {
        std::cerr << "usage: " << argv[0] << " MOTION.gguf TEXT_BUNDLE PROMPT.txt FRAMES STEPS SEED OUTPUT_DIR [NEGATIVE_PROMPT.txt]\n"
                  << "   or: " << argv[0] << " MOTION.gguf TEXT_BUNDLE --sequence TRANSITION STEPS SEED OUTPUT_DIR FRAME PROMPT.txt [FRAME PROMPT.txt ...]\n";
        return 2;
    }
    std::ifstream prompt_file(argv[3]);
    const std::string prompt{std::istreambuf_iterator<char>(prompt_file), {}};
    if (!prompt_file && prompt.empty()) throw std::runtime_error("cannot read prompt");
    const auto frames = static_cast<unsigned>(std::stoul(argv[4]));
    const auto steps = static_cast<unsigned>(std::stoul(argv[5]));
    const auto seed = static_cast<std::uint64_t>(std::stoull(argv[6]));
    // argv[8], when present, is an optional negative-prompt file — additive, so callers built
    // against the original 8-arg form (generate-motion.ps1) keep working unchanged.
    const std::string negative_prompt = argc == 9 ? read_file(argv[8]) : std::string{};
    auto model = kimodo::model::load(argv[1], argv[2]);
    if (!model) throw std::runtime_error(model.error());
    const char *constraint_path=std::getenv("KIMODO_CONSTRAINTS_FILE");
    auto constraints=constraint_path?read_constraints(constraint_path):std::vector<kimodo::pose_constraint>{};
    auto motion = constraints.empty()
        ? (*model)->generate_text(prompt, frames, steps, seed,
            env_float("KIMODO_TEXT_CFG", 2.F), env_float("KIMODO_CONSTRAINT_CFG", 2.F), negative_prompt)
        : (*model)->generate_text_constrained(prompt, frames, steps, seed,
            env_float("KIMODO_TEXT_CFG", 2.F), env_float("KIMODO_CONSTRAINT_CFG", 2.F),constraints,negative_prompt);
    if (!motion) throw std::runtime_error(motion.error());
    write_motion(argv[7], *motion);
    std::cout << "generated " << motion->frames << " frames with " << motion->joints << " joints\n";
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
