//this component was written by Jeffrey P. Spence in 2016
//adapted from table_gen written by Andrew H. Chan
// email: spence.jeffrey@berkeley.edu

#include "convert_table/convert_table_component.h"
#include "convert_table/convert_table_options.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <vector>
#include "common/read_confs.h"
#include "common/rho_finder.h"
#include "common/version_number.h"
#include "table_gen/output_writer.h"
#include "table_gen/table_management.h"

std::vector<std::string> split(std::string const &input) {
    std::istringstream buffer(input);
    std::vector<std::string> ret;
    
    std::copy(std::istream_iterator<std::string>(buffer),
              std::istream_iterator<std::string>(),
              std::back_inserter(ret));
    return ret;
}


int ConvertTableComponent(std::string const base_command, int argc, char **argv) {
    //uses the same format as table_gen
    uint64_t version_number = TABLE_GEN_VERSION_OUTPUT;
    uint64_t version_bit_string = TABLE_GEN_SALT + version_number;
    std::string version_string = TABLE_GEN_VERSION_STRING;
    CmdLineOptionsConvertTable cmd_line_options(base_command,
                                            argc,
                                            argv,
                                            version_string);
    if (!cmd_line_options.success()) {
        std::exit(1);
    }
    printf("%s\n", ShowCommandLineOptions(argc, argv).c_str());
    std::string const &input_file = cmd_line_options.input_file_;
    std::string const &output_file = cmd_line_options.output_file_;
    
    
    std::ifstream ldhatfile;
    ldhatfile.open(input_file);
    
    std::string curr_line;
    //read in header
    std::getline(ldhatfile, curr_line); //num configs, num haps, ignore
    std::getline(ldhatfile, curr_line, ' '); //number that does nothing
    std::getline(ldhatfile, curr_line); //theta
    std::vector<double> thetas;
    thetas.push_back(std::stod(curr_line));
    
    //get rhos
    std::getline(ldhatfile, curr_line);
    std::vector<std::string> rho_array = split(curr_line);
    std::vector<double> rhoArg;
    if (rho_array.size() == 2){ //this is an LDHat table
        rhoArg.push_back(0.0);  //LDHat tables always start at 0.0
        rhoArg.push_back(std::stod(rho_array[1]) / (std::stod(rho_array[0]) - 1));   //stepsize is max_rho / num_rhos
        rhoArg.push_back(std::stod(rho_array[1]));
    } else {                    //this is LDHelmet format table
        for(int i = 0; i < rho_array.size(); i++){
            rhoArg.push_back(std::stod(rho_array[i]));
        }
    }
    boost::tuple<std::vector<double>, std::vector<double> > rho_grid = ParseRhoRange(rhoArg);
    std::vector<double> rhos = GetRhoList(rho_grid);
    assert(rhos.size() > 0);
    if (rhos.front() != 0.0) {
        fprintf(stderr,
                "The rho values must begin at 0.0.\n");
        std::exit(1);
    }
    
    
    
    //Print out what we've read in, so far
    printf("theta: ");
    for (int i = 0; i < static_cast<int>(thetas.size()); ++i) {
        printf("%g ", thetas[i]);
    }
    printf("\n");
    
    printf("rho values: ");
    for (int i = 0; i < static_cast<int>(rhos.size()); ++i) {
        printf("%g ", rhos[i]);
    }
    printf("\n");
    
    
    
    //Read in all of the configs
    printf("Reading likelihoods from file.\n");
    std::vector<std::pair<Conf, std::vector<double> > > conf_list_master;
    while(std::getline(ldhatfile,curr_line)){
        std::vector<std::string> line_array = split(curr_line);
        //read in the number of the different types
        int aa = std::stoi(line_array[2]);
        int ab = std::stoi(line_array[3]);
        int ba = std::stoi(line_array[4]);
        int bb = std::stoi(line_array[5]);
        
        int rho_offset = 0;
        while (line_array[rho_offset].compare(":") != 0 && rho_offset < line_array.size()){
            rho_offset += 1;
        }
        if (rho_offset == line_array.size()){
            fprintf(stderr, "Config likelihood lines must contain ':' \n");
            std::exit(1);
        }
        if (rho_offset + rhos.size() + 1 != line_array.size()){
            fprintf(stderr, "Number of table entries does not match number of rhos \n");
            std::exit(1);
        }
        std::vector<double> rhoLogLiks;
        for(int rhoIdx = rho_offset + 1; rhoIdx < line_array.size(); ++rhoIdx)
        {
            double currLoglik = exp(std::stod(line_array[rhoIdx]));
            rhoLogLiks.push_back(currLoglik);
        }
        
        //undo the symmetries that LDHat assumes and LDHelmet does not
        Conf currConf1(0, 0, 0, 0, aa, ab, ba, bb);
        std::pair<Conf, std::vector<double> > toAdd1(currConf1, rhoLogLiks);
        conf_list_master.push_back(toAdd1);
        
        //swap allele at first locus
        Conf currConf2(0, 0, 0, 0, ba, bb, aa, ab);
        std::pair<Conf, std::vector<double> > toAdd2(currConf2, rhoLogLiks);
        conf_list_master.push_back(toAdd2);
        
        //swap allele at second locus
        Conf currConf3(0, 0, 0, 0, ab, aa, bb, ba);
        std::pair<Conf, std::vector<double> > toAdd3(currConf3, rhoLogLiks);
        conf_list_master.push_back(toAdd3);
        
        //swap allele at both loci
        Conf currConf4(0, 0, 0, 0, bb, ba, ab, aa);
        std::pair<Conf, std::vector<double> > toAdd4(currConf4, rhoLogLiks);
        conf_list_master.push_back(toAdd4);
        
        //swap loci
        Conf currConf5(0, 0, 0, 0, aa, ba, ab, bb);
        std::pair<Conf, std::vector<double> > toAdd5(currConf5, rhoLogLiks);
        conf_list_master.push_back(toAdd5);
        
        //swap loci, swap allele at first locus
        Conf currConf6(0, 0, 0, 0, ba, aa, bb, ab);
        std::pair<Conf, std::vector<double> > toAdd6(currConf6, rhoLogLiks);
        conf_list_master.push_back(toAdd6);
        
        //swap loci, swap allele at second locus
        Conf currConf7(0, 0, 0, 0, ab, bb, aa, ba);
        std::pair<Conf, std::vector<double> > toAdd7(currConf7, rhoLogLiks);
        conf_list_master.push_back(toAdd7);
        
        //swap loci, both alleles
        Conf currConf8(0, 0, 0, 0, bb, ab, ba, aa);
        std::pair<Conf, std::vector<double> > toAdd8(currConf8, rhoLogLiks);
        conf_list_master.push_back(toAdd8);
    }
    struct sort_pre {
        bool operator()(const std::pair<Conf,std::vector<double > > &left, const std::pair<Conf,std::vector<double > > &right) {
            return left.first < right.first;
        }
    };
    std::sort(conf_list_master.begin(), conf_list_master.end(), sort_pre());
    std::vector<Conf> conf_list;
    for(int i = 0; i < conf_list_master.size(); ++i){
        conf_list.push_back(conf_list_master[i].first);
    }
    std::vector<size_t> degree_seps;
    uint32_t max_degree, max_sample_size, max_locus;
    boost::tie(degree_seps, max_degree, max_sample_size, max_locus) =
                    PreprocessConfs(conf_list);
    
    printf("Largest sample size: %d.\n", static_cast<int>(max_sample_size));
    
    
    //Start writing:
    InputConfBinaryWriter input_conf_binary_writer(output_file,
                                                   conf_list,
                                                   degree_seps);
    {
        // Write input confs.
        InputConfBinaryWriter input_conf_binary_writer(output_file,
                                                       conf_list,
                                                       degree_seps);
        {
            // Write version number.
            int num_written;
            num_written = fwrite(reinterpret_cast<void const *>(&version_bit_string),
                                 sizeof(version_bit_string),
                                 1, input_conf_binary_writer.fp_);
            assert(num_written == 1);
            
            // Write num confs.
            uint64_t num_conf_list = conf_list.size();
            num_written = fwrite(reinterpret_cast<void const *>(&num_conf_list),
                                 sizeof(num_conf_list),
                                 1, input_conf_binary_writer.fp_);
            assert(num_written == 1);
            
            // Write theta.
            double theta = thetas.front();
            num_written = fwrite(reinterpret_cast<void const *>(&theta),
                                 sizeof(theta),
                                 1, input_conf_binary_writer.fp_);
            assert(num_written == 1);
            
            // Write number of rho segments, excluding end point.
            if (rho_grid.get<0>().size() == 0) {
                fprintf(stderr,
                        "Error: The size of rho_grid is 0.\n");
                std::exit(1);
            }
            uint64_t num_rho_segments = rho_grid.get<0>().size() - 1;
            num_written = fwrite(reinterpret_cast<void const *>(&num_rho_segments),
                                 sizeof(num_rho_segments),
                                 1, input_conf_binary_writer.fp_);
            assert(num_written == 1);
            
            // Write rho segments.
            assert(rho_grid.get<0>().size() > 0);
            for (uint32_t i = 0; i < rho_grid.get<0>().size() - 1; ++i) {
                double start = rho_grid.get<0>()[i];
                double delta = rho_grid.get<1>()[i];
                num_written = fwrite(reinterpret_cast<void const *>(&start),
                                     sizeof(start),
                                     1, input_conf_binary_writer.fp_);
                assert(num_written == 1);
                
                num_written = fwrite(reinterpret_cast<void const *>(&delta),
                                     sizeof(delta),
                                     1, input_conf_binary_writer.fp_);
                assert(num_written == 1);
            }
            num_written = fwrite(reinterpret_cast<void const *>(
                                                                &(rho_grid.get<0>().back())),
                                 sizeof(rho_grid.get<0>().back()),
                                 1, input_conf_binary_writer.fp_);
            assert(num_written == 1);
            
            // Write confs.
            for (std::vector<Conf>::const_iterator citer = conf_list.begin();
                 citer != conf_list.end();
                 ++citer) {
                Conf::BinaryRep::RepType rep = Conf::BinaryRep(*citer).rep();
                num_written = fwrite(reinterpret_cast<void const *>(&rep),
                                     sizeof(rep),
                                     1, input_conf_binary_writer.fp_);
                assert(num_written == 1);
            }
        }
        
        assert(thetas.size() == 1);
        for (size_t theta_id = 0; theta_id < thetas.size(); ++theta_id) {
            for (size_t rho_id = 0; rho_id < rhos.size(); ++rho_id) {
                printf("likelihoods written for rho: %g\n", rhos[rho_id]);
                
                Vec8 table(max_degree + 1);
                AllocateMemoryToTable(&table, max_degree);
                for (int confIdx = 0; confIdx < conf_list_master.size(); ++confIdx){
                    SetTable(&table, conf_list_master[confIdx].first, conf_list_master[confIdx].second[rho_id]);
                }
                input_conf_binary_writer.Write(max_degree, table);
            }
        }
    }
    return 0;
}