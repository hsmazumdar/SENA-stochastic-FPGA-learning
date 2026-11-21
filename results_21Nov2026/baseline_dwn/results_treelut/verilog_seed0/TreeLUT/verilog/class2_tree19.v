module class2_tree19(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if(((~i[123]) & (~i[8])) | (i[149] & i[123] & (~i[8])) | (i[421] & i[342] & i[187] & i[283] & i[8]) | ((~i[420]) & (~i[342]) & i[187] & i[283] & i[8]) | (i[340] & i[242] & (~i[187]) & i[283] & i[8]) | ((~i[340]) & i[242] & (~i[187]) & i[283] & i[8]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
