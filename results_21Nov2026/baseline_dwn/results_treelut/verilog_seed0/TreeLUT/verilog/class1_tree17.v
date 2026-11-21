module class1_tree17(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if(((~i[235]) & i[192]) | (i[234] & i[51] & (~i[192])) | (i[104] & i[86] & i[213] & i[235] & i[192]) | ((~i[63]) & (~i[86]) & i[213] & i[235] & i[192]) | (i[361] & i[337] & (~i[213]) & i[235] & i[192]) | ((~i[151]) & i[49] & (~i[234]) & i[51] & (~i[192])))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
